"""입출력 리뷰 시트(harness/io_review.py) 테스트 (LLM 호출 없음).

작은 가짜 run_stats(실 모델 1개, mock 1개)로 HTML과 마크다운이 생성되고,
입력 T점수·보호자 의견·모델명·블록 텍스트가 담기며, 보호자 어절이 인용된 자리에
하이라이트 마크업이 붙는지 확인한다. 현행 스키마는 prep 1블록(질문, ADR 0010과 그 보강)이고, 옛 4블록
(explain overview + prep counselor_briefing), 5블록(before_counseling), 2블록(observation_points) run_stats는
구 스키마로 표기해 읽는다.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

import io_review  # noqa: E402
from src.guardrails import TASK_BLOCKS  # noqa: E402

PROFILES = ROOT / "data" / "profiles"


def _run(model, mode="api", regen=False, tokens=True):
    """p2 프로파일에 대한 최소 run_stats 런 1건 (현행 1블록 스키마)."""
    prep_task = {
        "regen_count": 1 if regen else 0,
        "violations_by_attempt": {"attempt0": {"G8": 2}} if regen else {},
        "block_states": {"questions_for_counselor": "regen_pass" if regen else "pass"},
        "state_counts": {"pass": 0 if regen else 1, "regen_pass": 1 if regen else 0, "fallback": 0},
    }
    calls = [
        {"profile_id": "p2_partial_borderline", "task": "prep", "attempt": 0, "duration_s": 5.0,
         "prompt_tokens": 3400 if tokens else None, "completion_tokens": 600 if tokens else None},
    ]
    if regen:
        calls.append({"profile_id": "p2_partial_borderline", "task": "prep", "attempt": 1, "duration_s": 4.0,
                      "prompt_tokens": 3500 if tokens else None, "completion_tokens": 610 if tokens else None})
    return {
        "profile_id": "p2_partial_borderline",
        "model": model,
        "settings": {"reasoning_effort": "none", "num_ctx": 8192, "transport": "ollama-native", "temperature": 0.2}
        if mode == "api" else {},
        "tasks": {"prep": prep_task},
        "llm_calls": calls,
        "total_prompt_tokens": sum(c["prompt_tokens"] or 0 for c in calls),
        "total_completion_tokens": sum(c["completion_tokens"] or 0 for c in calls),
        "total_llm_seconds": round(sum(c["duration_s"] for c in calls), 2),
        "quality": {
            "jargon": {"term_hits": 3, "by_term": {"척도": 2, "준임상": 1}, "blocks_total": 2,
                       "blocks_with_term": 2, "glossed_blocks": 0, "residual_rate": 1.0, "gloss_rate": 0.0},
            "reflection": {"tokens": ["학원", "숙제", "놀이터", "또래"], "tokens_hit": ["학원", "숙제"],
                           "token_rate": 0.5, "items_total": 2, "items_reflected": 1, "item_rate": 0.5,
                           "questions_total": 2, "questions_reflected": 1},
            "direction_warnings": [],
        },
        "outputs": {
            "prep": {
                "questions_for_counselor": [
                    {"question": f"[{model}] 학원 숙제를 앞에 두면 딴 데를 보는 것은 어떻게 연결되나요?", "source_scale": "attention"},
                    {"question": f"[{model}] 위축 척도가 준임상인 것은 무엇부터 보게 되나요?", "source_scale": "withdrawn"},
                    {"question": "[안전 문구]", "source_scale": "total_problems", "_fallback": True},
                ],
            },
        },
    }


def _legacy_four_block_run(model):
    """옛 4블록 run_stats (explain overview + prep counselor_briefing, ADR 0009 시절). 관찰 포인트 블록(2블록 시절)도 포함."""
    run = _run(model)
    run["tasks"]["explain"] = {"regen_count": 0, "violations_by_attempt": {},
                               "block_states": {"overview": "pass"},
                               "state_counts": {"pass": 1, "regen_pass": 0, "fallback": 0}}
    run["tasks"]["prep"]["block_states"]["counselor_briefing"] = "pass"
    run["tasks"]["prep"]["block_states"]["observation_points"] = "pass"
    run["outputs"]["prep"]["observation_points"] = [
        {"point": f"[{model}] 숙제할 때 시선이 옮겨 가는 횟수 세어 두기", "source_scale": "attention"}]
    run["llm_calls"].insert(0, {"profile_id": "p2_partial_borderline", "task": "explain", "attempt": 0,
                                "duration_s": 2.5, "prompt_tokens": 2400, "completion_tokens": 230})
    run["outputs"]["explain"] = {
        "overview": f"[{model}] 학원 숙제를 앞에 두면 딴 데를 본다는 관찰은 주의집중 척도(T점수 67, 준임상)와 연결됩니다."}
    run["outputs"]["prep"]["counselor_briefing"] = f"[{model}] 종합지표: 총 문제행동 T=57(정상). 준임상 척도: 주의집중(T=67)."
    return run


def _write_stats(tmp_path, name, mode, runs):
    p = tmp_path / name / "run_stats.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"mode": mode, "runs": runs}, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def stats_files(tmp_path):
    a = _write_stats(tmp_path, "real", "api", [_run("gemma4:12b", regen=True)])
    b = _write_stats(tmp_path, "mock", "mock", [_run("mock", mode="mock", tokens=False)])
    return a, b


def test_html_contains_inputs_models_blocks_and_highlight(stats_files):
    runs = io_review.load_runs([str(p) for p in stats_files])
    assert [r["model"] for r in runs] == ["gemma4:12b", "mock"]
    html = io_review.build_review_html(runs, PROFILES, env_note="테스트 서버")

    # 입력: T점수와 밴드, 보호자 의견 원문, 상담까지 남은 날
    assert "T 67" in html and "준임상" in html and "정상" in html
    assert "총 문제행동" in html and "주의집중" in html
    assert "학원 숙제를 앞에 두면 딴 데를 자주 봅니다" in html.replace('<mark class="hit">', "").replace("</mark>", "") \
        .replace('<span class="cand">', "").replace("</span>", "")
    assert "상담까지 <b>5</b>일" in html
    # 출력: 모델명 열, mock 표기, 1블록 텍스트 (옛 블록 행은 없다)
    assert "gemma4:12b" in html and "mock 템플릿(실제 LLM 아님)" in html
    plain = html.replace('<mark class="hit">', "").replace("</mark>", "")
    for text in ("[gemma4:12b] 위축 척도가 준임상인 것은 무엇부터 보게 되나요?",
                 "[mock] 학원 숙제를 앞에 두면 딴 데를 보는 것은 어떻게 연결되나요?"):
        assert text in plain
    assert "1블록" in html and "구 스키마" not in html.split("<section")[1]
    assert "before_counseling" not in html and "counselor_briefing" not in html and "overview" not in html
    assert "observation_points" not in html and "가정 관찰 포인트" not in html
    assert "참고 척도: 주의집중" in html and "안전 문구" in html
    # 하이라이트: 인용된 어절에 mark, 인용되지 않은 어절은 점선 후보
    assert '<mark class="hit">학원</mark> <mark class="hit">숙제</mark>' in html
    assert '<span class="cand">놀이터</span>' in html
    # 메타: 재생성 후 통과 + 걸린 규칙과 사유, 토큰·시간, 실행 환경 메모
    assert "재생성 후 통과" in html and "G8" in html and "밴드 라벨 정합" in html
    assert "6,900" in html and "테스트 서버" in html
    # 부록: 시도별 원문이 없다는 사실을 적는다
    assert "시도별 원문이 저장되지 않습니다" in html and "원문 미저장" in html


def test_highlight_helpers_escape_and_prefer_longest_token():
    out = str(io_review.highlight_html("학원 숙제 <b>", ["학원", "학원 숙제"]))
    assert out == '<mark class="hit">학원 숙제</mark> &lt;b&gt;'
    out = str(io_review.highlight_html("놀이터에서 또래", ["놀이터", "또래"], hit={"또래"}))
    assert out == '<span class="cand">놀이터</span>에서 <mark class="hit">또래</mark>'
    assert io_review.highlight_md("a|b 학원", ["학원"]) == "a\\|b **학원**"


def test_markdown_and_cli(stats_files, tmp_path):
    out_html = tmp_path / "o" / "review.html"
    out_md = tmp_path / "o" / "review.md"
    rc = io_review.main([str(stats_files[0]), str(tmp_path / "mock" / "*.json"), "--profiles", str(PROFILES),
                         "--out", str(out_html), "--md", str(out_md)])
    assert rc == 0 and out_html.exists()
    md = out_md.read_text(encoding="utf-8")
    assert "| 종합 | 총 문제행동 | 57 | 정상 |" in md
    assert "| 하위 | 주의집중 | 67 | 준임상 |" in md
    assert "**학원** **숙제**" in md and "gemma4:12b" in md and "mock 템플릿(실제 LLM 아님)" in md
    assert "G8 2건(밴드 라벨 정합)" in md and "원문 미저장" in md


def test_missing_profile_still_renders_outputs(tmp_path):
    runs = io_review.load_runs([str(_write_stats(tmp_path, "x", "api", [_run("gemma4:12b")]))])
    html = io_review.build_review_html(runs, tmp_path / "no_profiles_here")
    assert "프로파일을 읽지 못함" in html
    assert "[gemma4:12b] 위축 척도가 준임상인 것은 무엇부터 보게 되나요?" in html
    # 프로파일이 없어도 run_stats의 어절 목록으로 표시한다
    assert '<mark class="hit">학원</mark>' in html


def test_blocks_match_guardrail_schema():
    """리뷰 시트의 블록 목록은 가드레일의 LLM 블록(2개)과 같은 집합이다 (스키마 드리프트 방지)."""
    assert [(t, b) for t, b, _l, _k in io_review.BLOCKS] == [(t, b) for t, bs in TASK_BLOCKS.items() for b in bs]
    assert len(io_review.BLOCKS) == 1
    assert set(io_review.LEGACY_BLOCKS) == {("explain", "overview"), ("explain", "before_counseling"),
                                            ("prep", "counselor_briefing"), ("prep", "observation_points")}
    assert set(io_review.RULE_KO) == {f"G{i}" for i in range(1, 13)}


def test_legacy_four_and_five_block_run_stats_still_render(tmp_path):
    """옛 4블록(overview, counselor_briefing), 5블록(before_counseling), 2블록(observation_points) run_stats를 넣어도 깨지지 않는다.

    옛 블록은 "구 스키마" 표기의 참고용 행으로 나오고, 어절 집계와 폴백 분모(x/1)에서는 빠진다.
    새 스키마 런과 섞이면 옛 블록이 없는 열은 "출력 없음"이다.
    """
    old = _legacy_four_block_run("gemma4:12b")
    old["outputs"]["explain"]["before_counseling"] = "[old] 놀이터에서 본 장면을 메모해 두시면 재료가 됩니다."
    old["tasks"]["explain"]["block_states"]["before_counseling"] = "pass"
    new = _run("mock", mode="mock", tokens=False)
    files = [_write_stats(tmp_path, "old", "api", [old]), _write_stats(tmp_path, "new", "mock", [new])]
    runs = io_review.load_runs([str(p) for p in files])

    html = io_review.build_review_html(runs, PROFILES)
    plain = html.replace('<mark class="hit">', "").replace("</mark>", "")
    assert "[old] 놀이터에서 본 장면을 메모해 두시면 재료가 됩니다." in plain
    assert "[gemma4:12b] 종합지표: 총 문제행동 T=57(정상)." in plain
    assert "보호자의 관찰과 검사 소견<small>overview · 구 스키마(결정론 조립으로 전환됨)</small>" in html
    assert "상담사용 사전 요약<small>counselor_briefing · 구 스키마(결정론 조립으로 전환됨)</small>" in html
    assert "상담 전 안내<small>before_counseling · 구 스키마(고정 문구로 전환됨)</small>" in html
    assert "가정 관찰 포인트<small>observation_points · 구 스키마(결정론 조립으로 전환됨)</small>" in html
    assert "[gemma4:12b] 숙제할 때 시선이 옮겨 가는 횟수 세어 두기" in plain      # 옛 목록 블록도 항목 단위로 그린다
    assert html.count('<tr class="legacy">') == 4

    sec = io_review.build_context(runs, PROFILES)["sections"][0]
    assert [r["block"] for r in sec["rows"] if not r["legacy"]] == [b for _t, b, _l, _k in io_review.BLOCKS]
    assert [(r["task"], r["block"]) for r in sec["rows"] if r["legacy"]] == [
        ("explain", "overview"), ("explain", "before_counseling"), ("prep", "counselor_briefing"),
        ("prep", "observation_points")]
    old_col, new_col = sec["columns"]
    assert old_col["cells"]["overview"]["kind"] == "text" and old_col["cells"]["counselor_briefing"]["kind"] == "text"
    assert old_col["cells"]["observation_points"]["kind"] == "list"
    assert new_col["cells"]["overview"]["kind"] == "missing"     # 새 스키마 런에는 없는 블록
    assert old_col["blocks_total"] == 1 and old_col["fallback_blocks"] == 0
    assert "놀이터" not in old_col["hit"]                                  # 옛 블록의 어절은 집계하지 않는다
    assert "놀이터" not in sec["hit_union"]
    assert old_col["task_meta"]["explain"]["attempts"] == 1               # 옛 explain 호출의 토큰·시간은 남긴다

    md = io_review.build_review_md(runs, PROFILES)
    assert "#### 보호자의 관찰과 검사 소견 (overview) - 구 스키마(결정론 조립으로 전환됨)" in md
    assert "#### 상담 전 안내 (before_counseling) - 구 스키마(고정 문구로 전환됨)" in md
    # 저장소에 커밋된 실측 산출물(examples/api)도 스키마 세대와 무관하게 읽힌다
    api = io_review.load_runs([str(ROOT / "examples" / "api" / "run_stats.json")])
    api_html = io_review.build_review_html(api, PROFILES)
    assert "gemma4:12b" in api_html and "p2_partial_borderline" in api_html


def test_bad_inputs_fail_cleanly(tmp_path, capsys):
    assert io_review.main([str(tmp_path / "none" / "*.json"), "--out", str(tmp_path / "r.html")]) == 1
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert io_review.main([str(bad), "--out", str(tmp_path / "r.html")]) == 1
    assert "리뷰 시트를 만들지 못했습니다" in capsys.readouterr().err
