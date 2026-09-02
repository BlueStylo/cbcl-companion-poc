"""LLM 실측 런의 입력과 출력을 나란히 놓고 읽는 리뷰 시트.

main.py(--api 또는 --mock)가 남기는 run_stats.json 여러 개와 프로파일 디렉토리를 받아,
프로파일별로 왼쪽에 입력(가상 아동 정보, T점수와 밴드, 보호자 의견 원문, 상담까지 남은 날)을,
오른쪽에 모델별 최종 출력 5블록을 격자로 놓은 정적 HTML 1개를 만든다. 보호자 의견의 어절이
출력 어디에 등장했는지 색으로 표시해 "입력이 출력에 어떻게 반영됐는가"를 한눈에 보는 것이 목적.

    python harness/io_review.py out/bench/*/run_stats.json --profiles data/profiles --out out/io_review.html
    python harness/io_review.py ... --md out/io_review.md          # 같은 내용의 마크다운도 함께
    python harness/io_review.py ... --env-note "별도 GPU 서버, Ollama 0.33.2"   # run_stats에 없는 환경 정보

한계: run_stats에는 시도별 원문이 없고 규칙별 위반 건수만 있다(src/generator.py summarize_run).
그래서 부록은 걸린 규칙과 통과한 최종 문장까지만 보여 준다. 반영률 수치는 하네스 정의대로
질문·관찰 항목만 세지만(src/quality.py), 색 표시는 5블록 전체에 한다.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from src.parser import BAND_KO, COMPOSITE_IDS, SCALE_NAMES, SYNDROME_IDS, ProfileError, load_profile
from src.quality import fmt_rate, note_tokens

TEMPLATES_DIR = ROOT / "src" / "templates"
TEMPLATE_NAME = "io_review.html.j2"

# 출력 5블록 (리포트에 나가는 순서): (task, block, 라벨, 목록이면 항목 텍스트 키)
BLOCKS = (
    ("explain", "overview", "보호자의 관찰과 검사 소견", None),
    ("explain", "before_counseling", "상담 전 마음가짐 안내", None),
    ("prep", "questions_for_counselor", "상담사에게 물어볼 질문", "question"),
    ("prep", "observation_points", "가정 관찰 포인트", "point"),
    ("prep", "counselor_briefing", "상담사용 사전 요약", None),
)
TASK_KO = {"explain": "해설 호출 (explain)", "prep": "상담 준비 호출 (prep)"}
# README '가드레일 규칙' 표의 검사명을 짧게
RULE_KO = {
    "G1": "진단명", "G2": "심각성 단정", "G3": "수치 대조", "G4": "근거 링크",
    "G5": "스키마", "G6": "처방·치료 권고", "G7": "형식 누출", "G8": "밴드 라벨 정합",
    "G9": "정상 척도 근거", "G10": "예시 오염",
}
STATE_KO = {"pass": "첫 시도 통과", "regen_pass": "재생성 후 통과", "fallback": "안전 문구 대체"}
MOCK_MODELS = {"mock", "mock-template"}
MOCK_LABEL = "mock 템플릿(실제 LLM 아님)"
SEX_KO = {"female": "여", "male": "남"}
NO_ATTEMPT_TEXT = ("run_stats에는 시도별 원문이 저장되지 않습니다 (규칙별 위반 건수만 기록). "
                   "차단된 첫 시도의 문장은 표시할 수 없고, 아래는 재생성 후 리포트에 나간 최종 문장입니다.")


# ---------------------------------------------------------------- 입력 읽기

def expand_paths(patterns: list[str]) -> list[Path]:
    """경로 목록을 펼친다. 글롭 문자가 있으면 정렬해 확장하고, 매칭이 없으면 오류."""
    files: list[Path] = []
    for pat in patterns:
        if any(ch in pat for ch in "*?["):
            matches = sorted(glob.glob(pat))
            if not matches:
                raise FileNotFoundError(f"글롭에 맞는 파일이 없음: {pat}")
            files.extend(Path(m) for m in matches)
        else:
            files.append(Path(pat))
    return files


def load_runs(paths: list[str | Path]) -> list[dict]:
    """run_stats.json 파일들을 읽어 런 목록으로 편다. 각 런에 _source, _mode를 붙인다."""
    runs: list[dict] = []
    for f in expand_paths([str(p) for p in paths]):
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "runs" not in data:
            raise ValueError(f"{f}: run_stats 형식이 아님 (runs 키 없음)")
        for r in data["runs"]:
            run = dict(r)
            run["_source"] = str(f)
            run["_mode"] = data.get("mode", "?")
            runs.append(run)
    return runs


def is_mock(run: dict) -> bool:
    return run.get("model") in MOCK_MODELS or run.get("_mode") == "mock"


# ---------------------------------------------------------------- 어절 표시

def _token_pattern(tokens) -> re.Pattern | None:
    toks = sorted({t for t in tokens if t}, key=len, reverse=True)
    return re.compile("|".join(re.escape(t) for t in toks)) if toks else None


def highlight_html(text: str, tokens, hit=None) -> Markup:
    """text 안의 어절 자리를 표시한 HTML (이스케이프 포함).

    hit이 None이면 tokens 전부를 <mark class="hit">로, 주어지면 hit에 든 어절만 mark.hit,
    나머지 후보 어절은 <span class="cand">(점선)로 감싼다. 긴 어절부터 매칭해 겹침을 피한다.
    """
    text = text or ""
    pat = _token_pattern(tokens)
    if pat is None:
        return Markup(html.escape(text))
    out: list[str] = []
    pos = 0
    for m in pat.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        word = html.escape(m.group(0))
        if hit is None or m.group(0) in hit:
            out.append(f'<mark class="hit">{word}</mark>')
        else:
            out.append(f'<span class="cand">{word}</span>')
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return Markup("".join(out))


def highlight_md(text: str, tokens) -> str:
    """마크다운용: 어절을 **굵게**. 표 셀에 넣을 수 있게 세로줄과 줄바꿈을 정리한다."""
    text = (text or "").replace("|", "\\|").replace("\n", " ")
    pat = _token_pattern(tokens)
    return pat.sub(lambda m: f"**{m.group(0)}**", text) if pat else text


def tokens_found(tokens, texts: list[str]) -> set[str]:
    joined = "\n".join(t for t in texts if t)
    return {t for t in tokens if t and t in joined}


# ---------------------------------------------------------------- 셀과 열 조립

def _cell(value, text_key):
    """출력 블록 값을 셀 구조로 바꾼다. kind = text / list / missing."""
    if value is None:
        return {"kind": "missing", "texts": []}
    if text_key is None:
        text = value.get("text", "") if isinstance(value, dict) else str(value)
        return {"kind": "text", "text": text, "texts": [text]}
    items = []
    for it in value if isinstance(value, list) else []:
        if not isinstance(it, dict):
            continue
        sid = it.get("source_scale")
        items.append({
            "text": str(it.get(text_key, "")),
            "source_ko": SCALE_NAMES.get(sid, sid or "?"),
            "fallback": bool(it.get("_fallback")),
        })
    return {"kind": "list", "items": items, "texts": [i["text"] for i in items]}


def _violations(tinfo: dict) -> list[dict]:
    out = []
    for attempt, rules in sorted((tinfo.get("violations_by_attempt") or {}).items()):
        out.append({
            "attempt": attempt,
            "attempt_no": int(re.sub(r"\D", "", attempt) or 0) + 1,
            "rules": [{"id": rid, "count": n, "ko": RULE_KO.get(rid, "")}
                      for rid, n in sorted(rules.items())],
        })
    return out


def build_column(run: dict, tokens) -> dict:
    """런 1건을 출력 격자의 열 1개로 바꾼다."""
    tasks = run.get("tasks") or {}
    outputs = run.get("outputs") or {}
    calls = run.get("llm_calls") or []
    col = {
        "model": run.get("model", "?"),
        "is_mock": is_mock(run),
        "source": run.get("_source", ""),
        "mode": run.get("_mode", "?"),
        "settings": run.get("settings") or {},
        "task_meta": {},
        "cells": {},
        "hit": set(),
    }
    for task in ("explain", "prep"):
        tinfo = tasks.get(task) or {}
        tcalls = [c for c in calls if c.get("task") == task]
        has_tokens = any(c.get("prompt_tokens") is not None for c in tcalls)
        col["task_meta"][task] = {
            "attempts": len(tcalls) or (int(tinfo.get("regen_count", 0)) + 1 if tinfo else 0),
            "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in tcalls) if has_tokens else None,
            "completion_tokens": sum(c.get("completion_tokens") or 0 for c in tcalls) if has_tokens else None,
            "seconds": round(sum(c.get("duration_s") or 0 for c in tcalls), 1) if tcalls else None,
            "regen_count": int(tinfo.get("regen_count", 0)),
            "violations": _violations(tinfo),
        }
    for task, block, _label, text_key in BLOCKS:
        cell = _cell((outputs.get(task) or {}).get(block), text_key)
        state = ((tasks.get(task) or {}).get("block_states") or {}).get(block)
        cell["state"] = state
        cell["state_ko"] = STATE_KO.get(state, "상태 기록 없음")
        cell["violations"] = col["task_meta"][task]["violations"] if state in ("regen_pass", "fallback") else []
        cell["ambiguous"] = False
        col["hit"] |= tokens_found(tokens, cell["texts"])
        col["cells"][block] = cell
    # 위반은 호출(task) 단위 집계라, 같은 호출에서 통과 못 한 블록이 둘 이상이면 어느 블록의 위반인지 구분되지 않는다
    for task in ("explain", "prep"):
        non_pass = [b for t, b, _l, _k in BLOCKS if t == task and col["cells"][b]["state"] in ("regen_pass", "fallback")]
        for b in non_pass:
            col["cells"][b]["ambiguous"] = len(non_pass) > 1
    states = [c["state"] for c in col["cells"].values()]
    col["regen_total"] = sum(m["regen_count"] for m in col["task_meta"].values())
    col["fallback_blocks"] = states.count("fallback")
    col["blocks_total"] = len(BLOCKS)
    col["prompt_tokens"] = run.get("total_prompt_tokens")
    col["completion_tokens"] = run.get("total_completion_tokens")
    col["seconds"] = run.get("total_llm_seconds")
    col["quality"] = run.get("quality") or {}
    return col


def _fmt_tokens(pt, ct) -> str:
    if pt is None and ct is None:
        return "없음"
    return f"{pt or 0:,} / {ct or 0:,}"


def quality_rows(columns: list[dict], tokens) -> list[dict]:
    """섹션 끝의 모델별 품질 지표 표. 값은 run_stats의 quality를 그대로 옮긴다."""
    def cell(col, fn):
        try:
            return fn(col)
        except (KeyError, TypeError, ZeroDivisionError):
            return "-"

    def refl(col):
        return col["quality"]["reflection"]

    def jar(col):
        return col["quality"]["jargon"]

    specs = [
        ("표현 반영 (항목)", lambda c: f"{refl(c)['items_reflected']}/{refl(c)['items_total']} ({fmt_rate(refl(c)['item_rate'])})"),
        ("표현 반영 (어절)", lambda c: f"{len(refl(c)['tokens_hit'])}/{len(refl(c)['tokens'])} ({fmt_rate(refl(c)['token_rate'])})"),
        ("질문·관찰에서 놓친 어절", lambda c: ", ".join(t for t in refl(c)["tokens"] if t not in set(refl(c)["tokens_hit"])) or "없음"),
        ("5블록 어디에도 없는 어절", lambda c: ", ".join(t for t in tokens if t not in c["hit"]) or "없음"),
        ("용어 잔존", lambda c: f"{jar(c)['term_hits']}회 (용어 블록 {jar(c)['blocks_with_term']}/{jar(c)['blocks_total']}, 풀이 동반 {fmt_rate(jar(c)['gloss_rate'])})"),
        ("상위 용어", lambda c: ", ".join(f"{k} {v}" for k, v in list(jar(c)["by_term"].items())[:4]) or "없음"),
        ("방향 경고", lambda c: f"{len(c['quality']['direction_warnings'])}건"
         + ("".join(f" ({w['matched']})" for w in c["quality"]["direction_warnings"]) if c["quality"]["direction_warnings"] else "")),
        ("재생성", lambda c: f"{c['regen_total']}회"),
        ("폴백 블록", lambda c: f"{c['fallback_blocks']}/{c['blocks_total']}"),
        ("토큰 입력 / 출력", lambda c: _fmt_tokens(c["prompt_tokens"], c["completion_tokens"])),
        ("LLM 시간", lambda c: f"{c['seconds']}초" if c["seconds"] is not None else "-"),
    ]
    return [{"label": label, "values": [cell(c, fn) for c in columns]} for label, fn in specs]


# ---------------------------------------------------------------- 섹션과 문서

def _profile_input(profile, profile_path: Path, error: str | None) -> dict:
    if profile is None:
        return {"ok": False, "error": error or f"프로파일 파일을 찾지 못함: {profile_path}", "notes": [],
                "composites": [], "syndromes": []}
    child = profile.child
    scale = profile.scale_map()

    def row(sid):
        s = scale[sid]
        return {"name": s.name_ko, "t": s.t_score, "band": s.band, "band_ko": BAND_KO[s.band]}

    return {
        "ok": True,
        "alias": child.alias,
        "sex_ko": SEX_KO.get(child.sex, child.sex),
        "age": f"{child.age_years}세 {child.age_months}개월",
        "norm_group": child.norm_group,
        "test_date": profile.test_date,
        "instrument": profile.instrument,
        "composites": [row(sid) for sid in COMPOSITE_IDS],
        "syndromes": [row(sid) for sid in SYNDROME_IDS],
        "notes": list(profile.caregiver_notes),
        "days": profile.days_until_counseling,
        "scheduled": profile.counseling_scheduled,
        "criteria": {k: (v.normal_max_t, v.borderline_max_t) for k, v in profile.band_criteria.items()},
    }


def build_section(profile_id: str, runs: list[dict], profiles_dir: Path, model_order: list[str]) -> dict:
    profile_path = Path(profiles_dir) / f"{profile_id}.json"
    profile, error = None, None
    try:
        profile = load_profile(profile_path)
    except ProfileError as e:
        error = f"프로파일을 읽지 못함: {'; '.join(e.errors)}"
    inp = _profile_input(profile, profile_path, error)

    # 어절 후보: run_stats의 quality.reflection.tokens가 있으면 그것(하네스가 잰 것과 동일), 없으면 같은 휴리스틱으로 직접 추출
    tokens: list[str] = []
    for r in runs:
        t = ((r.get("quality") or {}).get("reflection") or {}).get("tokens")
        if t:
            tokens = list(t)
            break
    if not tokens and inp["notes"]:
        tokens = note_tokens(inp["notes"])

    ordered = sorted(runs, key=lambda r: model_order.index(r.get("model", "?")))
    columns = [build_column(r, tokens) for r in ordered]
    seen: dict[str, int] = {}
    for col in columns:
        seen[col["model"]] = seen.get(col["model"], 0) + 1
        col["label"] = col["model"] + (f" ({seen[col['model']]}회차)" if seen[col["model"]] > 1 else "")
    hit_union: set[str] = set()
    for col in columns:
        hit_union |= col["hit"]
    return {
        "profile_id": profile_id,
        "input": inp,
        "tokens": tokens,
        "hit_union": hit_union,
        "columns": columns,
        "quality_rows": quality_rows(columns, tokens),
    }


def build_context(runs: list[dict], profiles_dir: str | Path, env_note: str = "", title: str = "") -> dict:
    """HTML과 마크다운이 공유하는 문서 컨텍스트."""
    if not runs:
        raise ValueError("런이 없습니다 (run_stats.json의 runs가 비어 있음)")
    model_order: list[str] = []
    profile_order: list[str] = []
    for r in runs:
        if r.get("model", "?") not in model_order:
            model_order.append(r.get("model", "?"))
        if r.get("profile_id", "?") not in profile_order:
            profile_order.append(r.get("profile_id", "?"))
    setting_keys: list[str] = []
    for r in runs:
        for k in (r.get("settings") or {}):
            if k not in setting_keys:
                setting_keys.append(k)
    sources = [r.get("_source", "") for r in runs]
    try:
        # 준 경로 그대로 (상대 경로로 부르면 상대 경로로 남아 문서에 개인 디렉토리가 찍히지 않는다)
        source_root = os.path.commonpath([str(Path(x).parent) for x in sources if x]) if sources else ""
    except ValueError:  # 절대·상대 경로나 드라이브가 섞임
        source_root = ""
    run_rows = []
    for r in runs:
        tasks = r.get("tasks") or {}
        run_rows.append({
            "model": r.get("model", "?"),
            "is_mock": is_mock(r),
            "profile_id": r.get("profile_id", "?"),
            "mode": r.get("_mode", "?"),
            "settings": [str((r.get("settings") or {}).get(k, "-")) for k in setting_keys],
            "calls": len(r.get("llm_calls") or []),
            "regen": sum(int(t.get("regen_count", 0)) for t in tasks.values()),
            "fallback": sum(int((t.get("state_counts") or {}).get("fallback", 0)) for t in tasks.values()),
            "source": r.get("_source", ""),
            "source_short": "/".join(Path(r.get("_source", "")).parts[-2:]),
        })
    sections = [
        build_section(pid, [r for r in runs if r.get("profile_id") == pid], Path(profiles_dir), model_order)
        for pid in profile_order
    ]
    appendix = []
    for sec in sections:
        for col in sec["columns"]:
            for task, meta in col["task_meta"].items():
                if not meta["violations"]:
                    continue
                blocks = [(label, col["cells"][block]) for t, block, label, _k in BLOCKS
                          if t == task and col["cells"][block]["state"] in ("regen_pass", "fallback")]
                appendix.append({"profile_id": sec["profile_id"], "model": col["label"], "is_mock": col["is_mock"],
                                 "task": task, "meta": meta, "blocks": blocks, "tokens": sec["tokens"]})
    return {
        "title": title or "LLM 입출력 리뷰 시트",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "env_note": env_note,
        "n_runs": len(runs), "n_profiles": len(profile_order), "n_models": len(model_order),
        "model_order": model_order,
        "setting_keys": setting_keys,
        "run_rows": run_rows,
        "source_root": source_root,
        "sections": sections,
        "blocks": BLOCKS,
        "task_ko": TASK_KO,
        "appendix": appendix,
        "no_attempt_text": NO_ATTEMPT_TEXT,
        "mock_label": MOCK_LABEL,
    }


def build_review_html(runs: list[dict], profiles_dir: str | Path, env_note: str = "", title: str = "") -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape(["html", "j2"]))
    env.filters["hl"] = highlight_html
    env.filters["fmt_tokens"] = _fmt_tokens
    return env.get_template(TEMPLATE_NAME).render(**build_context(runs, profiles_dir, env_note, title))


def build_review_md(runs: list[dict], profiles_dir: str | Path, env_note: str = "", title: str = "") -> str:
    """같은 내용의 마크다운 (표 위주). 어절 표시는 굵게."""
    ctx = build_context(runs, profiles_dir, env_note, title)
    L: list[str] = []
    L.append(f"# {ctx['title']}")
    L.append("")
    L.append(f"생성 {ctx['generated_at']}, 런 {ctx['n_runs']}개, 프로파일 {ctx['n_profiles']}종, 모델 {ctx['n_models']}종. "
             "**굵게** 표시한 말은 보호자 의견의 어절이 출력에 등장한 자리입니다. "
             "반영률 수치는 하네스 정의대로 질문과 관찰 항목만 셉니다.")
    L.append("")
    L.append("## 포함된 런")
    L.append("")
    head = ["모델", "프로파일", "모드"] + ctx["setting_keys"] + ["호출", "재생성", "폴백 블록", "파일"]
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))
    for r in ctx["run_rows"]:
        model = r["model"] + (f" ({MOCK_LABEL})" if r["is_mock"] else "")
        L.append("| " + " | ".join([model, r["profile_id"], r["mode"]] + r["settings"]
                                   + [str(r["calls"]), str(r["regen"]), str(r["fallback"]), f"`{r['source_short']}`"]) + " |")
    L.append("")
    if ctx["source_root"]:
        L.append(f"파일 기준 디렉토리: `{ctx['source_root']}`")
        L.append("")
    if ctx["env_note"]:
        L.append(f"실행 환경 메모: {ctx['env_note']}")
        L.append("")
    for sec in ctx["sections"]:
        inp = sec["input"]
        tokens = sec["tokens"]
        L.append(f"## {sec['profile_id']}")
        L.append("")
        L.append("### 입력")
        L.append("")
        if inp["ok"]:
            L.append(f"- 아동(가상): {inp['alias']}, {inp['sex_ko']}, {inp['age']}, 규준 {inp['norm_group']}, "
                     f"검사일 {inp['test_date']}, {inp['instrument']}")
            L.append(f"- 상담까지 {inp['days']}일" + ("" if inp["scheduled"] else " (예약 안 됨)"))
            L.append("")
            L.append("| 구분 | 척도 | T점수 | 밴드 |")
            L.append("|---|---|---|---|")
            for row in inp["composites"]:
                L.append(f"| 종합 | {row['name']} | {row['t']} | {row['band_ko']} |")
            for row in inp["syndromes"]:
                L.append(f"| 하위 | {row['name']} | {row['t']} | {row['band_ko']} |")
            L.append("")
            L.append("보호자 의견 (원문):")
            L.append("")
            for i, n in enumerate(inp["notes"], 1):
                L.append(f"{i}. {highlight_md(n, sec['hit_union'])}")
            L.append("")
            missed = [t for t in tokens if t not in sec["hit_union"]]
            L.append(f"반영률 측정 대상 어절 {len(tokens)}개: {', '.join(tokens) or '없음'}"
                     + (f" (어느 모델 출력에도 없는 어절: {', '.join(missed)})" if missed else ""))
        else:
            L.append(f"- {inp['error']}")
        L.append("")
        L.append("### 출력 (모델별)")
        L.append("")
        for task, block, label, text_key in BLOCKS:
            L.append(f"#### {label} ({block})")
            L.append("")
            L.append("| 모델 | 출력 | 메타 |")
            L.append("|---|---|---|")
            for col in sec["columns"]:
                cell = col["cells"][block]
                meta = col["task_meta"][task]
                if cell["kind"] == "text":
                    body = highlight_md(cell["text"], tokens)
                elif cell["kind"] == "list":
                    body = "<br>".join(
                        f"{i}) {highlight_md(it['text'], tokens)} [근거: {it['source_ko']}]" + (" [안전 문구]" if it["fallback"] else "")
                        for i, it in enumerate(cell["items"], 1))
                else:
                    body = "(출력 없음)"
                m = [cell["state_ko"]]
                for v in cell["violations"]:
                    m.append(f"{v['attempt_no']}차 시도 위반: " + ", ".join(f"{r['id']} {r['count']}건({r['ko']})" for r in v["rules"]))
                m.append(f"호출 {meta['attempts']}회, 토큰 {_fmt_tokens(meta['prompt_tokens'], meta['completion_tokens'])}"
                         + (f", {meta['seconds']}초" if meta["seconds"] is not None else ""))
                label_m = col["label"] + (f" ({MOCK_LABEL})" if col["is_mock"] else "")
                L.append(f"| {label_m} | {body} | {'<br>'.join(m)} |")
            L.append("")
        L.append("### 품질 지표")
        L.append("")
        L.append("| 지표 | " + " | ".join(c["label"] for c in sec["columns"]) + " |")
        L.append("|---|" + "---|" * len(sec["columns"]))
        for row in sec["quality_rows"]:
            L.append(f"| {row['label']} | " + " | ".join(str(v).replace("|", "\\|") for v in row["values"]) + " |")
        L.append("")
    L.append("## 부록: 규칙에 걸린 시도")
    L.append("")
    if not ctx["appendix"]:
        L.append("규칙에 걸린 시도가 없습니다.")
    else:
        L.append(NO_ATTEMPT_TEXT)
        L.append("")
        for a in ctx["appendix"]:
            rules = "; ".join(f"{v['attempt_no']}차 시도: " + ", ".join(f"{r['id']} {r['count']}건({r['ko']})" for r in v["rules"])
                              for v in a["meta"]["violations"])
            L.append(f"### {a['profile_id']}, {a['model']}, {TASK_KO[a['task']]}")
            L.append("")
            L.append(f"- 걸린 규칙: {rules}")
            L.append(f"- 호출 {a['meta']['attempts']}회, 토큰 {_fmt_tokens(a['meta']['prompt_tokens'], a['meta']['completion_tokens'])}")
            L.append("")
            L.append("| 블록 | 차단된 첫 시도 | 통과한 재생성 (최종) |")
            L.append("|---|---|---|")
            for label, cell in a["blocks"]:
                if cell["kind"] == "text":
                    body = highlight_md(cell["text"], a["tokens"])
                elif cell["kind"] == "list":
                    body = "<br>".join(f"{i}) {highlight_md(it['text'], a['tokens'])}" for i, it in enumerate(cell["items"], 1))
                else:
                    body = "(출력 없음)"
                L.append(f"| {label} ({cell['state_ko']}) | 원문 미저장 | {body} |")
            L.append("")
    return "\n".join(L).rstrip() + "\n"


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LLM 실측 런의 입력과 출력을 나란히 보는 리뷰 시트 생성")
    ap.add_argument("run_stats", nargs="+", help="run_stats.json 경로 (여러 개, 글롭 허용)")
    ap.add_argument("--profiles", default=str(ROOT / "data" / "profiles"), help="프로파일 JSON 디렉토리 (기본 data/profiles)")
    ap.add_argument("--out", default="out/io_review.html", help="출력 HTML 경로 (기본 out/io_review.html)")
    ap.add_argument("--md", help="같은 내용의 마크다운도 이 경로에 쓴다")
    ap.add_argument("--env-note", default="", help="run_stats에 없는 실행 환경 메모 (서버, Ollama 버전 등)")
    ap.add_argument("--title", default="", help="문서 제목 (기본 'LLM 입출력 리뷰 시트')")
    args = ap.parse_args(argv)

    try:
        runs = load_runs(args.run_stats)
        html_text = build_review_html(runs, args.profiles, args.env_note, args.title)
        md_text = build_review_md(runs, args.profiles, args.env_note, args.title) if args.md else None
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"리뷰 시트를 만들지 못했습니다: {e}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    print(f"리뷰 시트 생성: {out} (런 {len(runs)}개)")
    if md_text is not None:
        md = Path(args.md)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(md_text, encoding="utf-8")
        print(f"마크다운 생성: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
