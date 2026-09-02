"""평가자용 탐색 콘솔 (Streamlit).

심사자가 JSON 파일 없이 슬라이더와 텍스트로 프로파일을 바꿔 가며 (1) 위계·밴드
시각화 (2) 보호자 문장을 인용하는 질문 생성 (3) 가드레일 동작을 확인한다.
렌더러·규칙·생성기는 기존 파이프라인 모듈을 그대로 호출한다 (두 벌 금지):
파서 → 위기 게이트 → generator → guardrails → report_html.

실행:  streamlit run app/explorer.py
쿼리:  ?example=p2_partial_borderline&autorun=1  (예시 로드 + mock 1회 자동 실행, 스크린샷용)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # streamlit 의존성 (표·막대 그래프)
import streamlit as st
import streamlit.components.v1 as components

from app.profile_builder import (COMPOSITE_NOTE, EXAMPLE_LABELS, EXAMPLE_ORDER, T_SLIDER_MAX,
                                 T_SLIDER_MIN, ExplorerInputs, band_for, build_profile,
                                 build_profile_raw, composite_hints, example_inputs)
from main import llm_failure_types, load_env_file
from src.generator import RULE_HINTS, generate_all, summarize_run
from src.guardrails import detect_crisis_signals
from src.llm_client import SEED_RULES, OpenAICompatClient, TemplateMockClient
from src.parser import BAND_KO, COMPOSITE_IDS, SCALE_NAMES, SYNDROME_IDS, ProfileError
from src.quality import fmt_rate
from src.report_html import build_crisis_html, build_pending_report_html, build_report_html

DEFAULT_EXAMPLE = "p2_partial_borderline"
load_env_file(ROOT / ".env")  # 시작 시 한 번: Ollama 모드 입력칸(base_url, 모델명)이 .env 값으로 미리 채워지도록
OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"
OLLAMA_DEFAULT_MODEL = "gemma4:12b"
API_DEFAULT_MODEL = "gpt-5.6-luna"  # src/llm_client.py 기본값과 동일하게 유지
MODES = {
    "mock": "mock 템플릿 (실제 LLM 생성이 아님)",
    "ollama": f"Ollama ({OLLAMA_DEFAULT_MODEL} 기본, 모델명 변경 가능)",
    "api": "API (OpenAI 호환 · .env의 LLM_* 사용)",
}
NOTE_SLOTS = 3
REPORT_HEIGHT = 980
CHIP_COLORS = {"normal": ("#eef2ec", "#4f6b52"), "borderline": ("#f8ecd9", "#8a6425"),
               "clinical": ("#f6e0d8", "#97462a")}
ACCENT = "#4a6fa5"  # 리포트 템플릿의 --accent와 같은 파랑. 결과 패널 표식에만 쓴다.
# 입력(왼쪽)·결과(오른쪽) 패널 구분. 셀렉터는 st.container(key=...)가 붙이는 st-key-* 클래스와
# 이 파일이 직접 출력하는 cb-* 클래스만 쓴다 (data-testid 등 내부 DOM에는 의존하지 않음).
PANEL_CSS = f"""
<style>
.st-key-input_panel {{ background: rgba(120, 130, 150, 0.07); border-color: rgba(120, 130, 150, 0.35) !important; }}
.st-key-result_panel {{ border-color: rgba(74, 111, 165, 0.35) !important; }}
.cb-badge {{ display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px;
             border-radius: 50%; font-size: 12px; font-weight: 700; color: #fff; background: #6b7280; flex: none; }}
.cb-badge.out {{ background: {ACCENT}; }}
.cb-guide {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; font-size: 0.9rem; opacity: 0.85;
             margin: -6px 0 14px; }}
.cb-head {{ display: flex; align-items: center; gap: 10px; padding-bottom: 10px; margin-bottom: 8px;
            border-bottom: 1px solid rgba(120, 130, 150, 0.3); }}
.cb-head .title {{ font-size: 1.15rem; font-weight: 700; }}
.cb-head .sub {{ font-size: 0.85rem; opacity: 0.65; }}
.cb-sec {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 8px; margin: 18px 0 6px; line-height: 1.3; }}
.cb-sec .title {{ font-weight: 700; }}
.cb-sec .note {{ font-size: 0.8rem; opacity: 0.65; }}
.cb-sec.in .idx {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; opacity: 0.5;
                   font-variant-numeric: tabular-nums; }}
.cb-sec.out {{ border-left: 3px solid {ACCENT}; padding-left: 10px; }}
</style>
"""


# ---------------------------------------------------------------- 상태

def _apply_inputs(inputs: ExplorerInputs) -> None:
    for sid, t in inputs.t_scores.items():
        st.session_state[f"t_{sid}"] = int(t)
    st.session_state["alias"] = inputs.alias
    st.session_state["sex"] = inputs.sex
    st.session_state["age_years"] = int(inputs.age_years)
    st.session_state["days"] = int(inputs.days_until_counseling)
    st.session_state["test_date"] = inputs.test_date
    for i in range(NOTE_SLOTS):
        st.session_state[f"note_{i}"] = inputs.notes[i] if i < len(inputs.notes) else ""


def _load_example() -> None:
    _apply_inputs(example_inputs(st.session_state["example"]))
    st.session_state.pop("run", None)


def _init_state() -> None:
    if st.session_state.get("initialized"):
        return
    st.session_state["initialized"] = True
    example = st.query_params.get("example", DEFAULT_EXAMPLE)
    st.session_state["example"] = example if example in EXAMPLE_ORDER else DEFAULT_EXAMPLE
    _apply_inputs(example_inputs(st.session_state["example"]))
    st.session_state["autorun"] = st.query_params.get("autorun") in ("1", "true")


def _collect_inputs() -> ExplorerInputs:
    return ExplorerInputs(
        t_scores={sid: int(st.session_state[f"t_{sid}"]) for sid in (*COMPOSITE_IDS, *SYNDROME_IDS)},
        notes=[st.session_state[f"note_{i}"] for i in range(NOTE_SLOTS)],
        alias=st.session_state["alias"],
        sex=st.session_state["sex"],
        age_years=int(st.session_state["age_years"]),
        days_until_counseling=int(st.session_state["days"]),
        test_date=st.session_state.get("test_date", ""),
    )


# ---------------------------------------------------------------- 파이프라인 (기존 모듈 호출)

def _make_client(mode: str, opts: dict):
    if mode == "mock":
        return TemplateMockClient(seed_rules=opts["seeds"], persist_seed=opts["persist"]), "mock-template"
    load_env_file(ROOT / ".env")
    if mode == "ollama":
        return OpenAICompatClient(base_url=opts["ollama_url"], api_key="ollama",
                                  model=opts["ollama_model"]), "api"
    return OpenAICompatClient(model=os.environ.get("LLM_MODEL") or API_DEFAULT_MODEL), "api"


def _run_pipeline(profile, mode: str, opts: dict, fingerprint: str) -> dict:
    """위기 게이트 → generator(가드레일 루프 포함) → 리포트. main.py와 같은 순서."""
    crisis = detect_crisis_signals(profile)
    if crisis:
        return {"fingerprint": fingerprint, "mode": mode, "crisis": crisis,
                "html": build_crisis_html(profile), "stats": None, "violations": []}
    client, mode_label = _make_client(mode, opts)
    results = generate_all(profile, client)
    stats = summarize_run(profile, results, client)
    violations = [{"task": task, "attempt": v.attempt, "rule": v.rule_id, "block": v.block, "matched": v.matched}
                  for task, r in results.items() for v in r.violations]
    return {"fingerprint": fingerprint, "mode": mode, "crisis": [],
            "html": build_report_html(profile, results, mode_label, getattr(client, "model", "")),
            "stats": stats, "violations": violations}


# ---------------------------------------------------------------- 화면 조각

def _chip(band: str) -> str:
    bg, fg = CHIP_COLORS[band]
    return (f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;'
            f'font-weight:700;background:{bg};color:{fg};white-space:nowrap">{BAND_KO[band]}</span>')


def _badge(kind: str, n: int) -> str:
    return f'<span class="cb-badge {kind}">{n}</span>'


def _panel_head(kind: str, n: int, title: str, sub: str) -> None:
    """열 맨 위의 패널 라벨. 두 열 모두 테두리 컨테이너 첫 요소라 같은 높이에 놓인다."""
    st.markdown(f'<div class="cb-head">{_badge(kind, n)}<span class="title">{title}</span>'
                f'<span class="sub">{sub}</span></div>', unsafe_allow_html=True)


def _section(kind: str, title: str, note: str = "", idx: str = "") -> None:
    """소제목. 입력 패널(in)은 회색 번호 캡션, 결과 패널(out)은 파란 좌측 보더로 어느 영역인지 표시."""
    idx_html = f'<span class="idx">{idx}</span>' if idx else ""
    note_html = f'<span class="note">{note}</span>' if note else ""
    st.markdown(f'<div class="cb-sec {kind}">{idx_html}<span class="title">{title}</span>{note_html}</div>',
                unsafe_allow_html=True)


def _slider_row(sid: str, criteria: dict) -> None:
    c1, c2 = st.columns([5, 1.4], vertical_alignment="bottom")
    c1.slider(SCALE_NAMES[sid], T_SLIDER_MIN, T_SLIDER_MAX, key=f"t_{sid}")
    c2.markdown(_chip(band_for(sid, int(st.session_state[f"t_{sid}"]), criteria)), unsafe_allow_html=True)


def _left_panel() -> tuple[bool, str, dict]:
    st.selectbox("예시 불러오기 (data/profiles 8종 → 슬라이더에 로드)", EXAMPLE_ORDER,
                 format_func=EXAMPLE_LABELS.get, key="example", on_change=_load_example)
    criteria = example_inputs(st.session_state["example"]).criteria

    _section("in", "하위 척도 T점수", "준임상 60~69 · 임상 70 이상", idx="01")
    for sid in SYNDROME_IDS:
        _slider_row(sid, criteria)

    _section("in", "종합 지표 T점수 (직접 입력)", "준임상 60~62 · 임상 63 이상", idx="02")
    st.caption(COMPOSITE_NOTE)
    for sid in COMPOSITE_IDS:
        _slider_row(sid, criteria)
    t_scores = {sid: int(st.session_state[f"t_{sid}"]) for sid in (*COMPOSITE_IDS, *SYNDROME_IDS)}
    for hint in composite_hints(t_scores, criteria):
        st.caption(f"참고 · {hint}")

    with st.expander("아동 정보 (최소) · 명백한 가상명만", expanded=False):
        st.text_input("가상 이름", key="alias")
        c1, c2 = st.columns(2)
        c1.radio("성별", ["female", "male"], format_func={"female": "여", "male": "남"}.get,
                 key="sex", horizontal=True)
        c2.number_input("만 나이", 4, 18, key="age_years")

    _section("in", "보호자 의견", "질문과 관찰 포인트가 이 문장을 인용합니다", idx="03")
    for i in range(NOTE_SLOTS):
        st.text_input(f"의견 {i + 1}", key=f"note_{i}", label_visibility="collapsed",
                      placeholder=f"보호자 의견 {i + 1} (비우면 제외)")
    st.number_input("상담까지 남은 날 (days_until_counseling)", 0, 60, key="days")

    _section("in", "생성 모드", "mock · Ollama · API", idx="04")
    mode = st.radio("생성 모드", list(MODES), format_func=MODES.get, key="mode", label_visibility="collapsed")
    opts: dict = {}
    if mode == "mock":
        st.caption("mock 템플릿 - 실제 LLM 생성이 아닙니다. 보호자 의견을 「」로 인용하고 준임상 이상 척도만 "
                   "근거로 삼는 규칙으로 조립한 응답이 같은 가드레일 루프를 통과합니다.")
        opts["seeds"] = set(st.multiselect("위반 시드 주입 (가드레일 확인용, mock 전용)", list(SEED_RULES),
                                           format_func=lambda r: f"{r} · {RULE_HINTS[r]}", key="seeds"))
        opts["persist"] = st.checkbox("재생성해도 계속 위반 (안전 문구 폴백까지 확인)", key="persist")
        if opts["seeds"]:
            st.caption("시드는 첫 시도 출력에 들어갑니다. G9는 정상 척도와 준임상 이상 척도가 함께 있을 때, "
                       "G10은 의견에 '학원 숙제'가 없을 때만 실제 위반이 됩니다.")
    elif mode == "ollama":
        env_url = os.environ.get("LLM_BASE_URL", "")
        env_model = os.environ.get("LLM_MODEL", "")
        opts["ollama_url"] = st.text_input("base_url", value=env_url if "11434" in env_url else OLLAMA_DEFAULT_URL,
                                           key="ollama_url")
        opts["ollama_model"] = st.text_input("모델명", value=env_model if "11434" in env_url and env_model
                                             else OLLAMA_DEFAULT_MODEL, key="ollama_model")
        st.caption(f"`ollama pull {OLLAMA_DEFAULT_MODEL}` 후 Ollama가 떠 있어야 합니다. 호출 1건 상한은 "
                   "LLM_TIMEOUT_S(기본 180초)이며 초과 시 리포트를 교체하지 않습니다 (fail-closed). "
                   "로컬 12B 모델은 재생성 포함 건당 수 분이 걸릴 수 있습니다.")
    else:
        has_key = bool(os.environ.get("LLM_API_KEY")) or (ROOT / ".env").exists()
        st.caption(f"환경변수 LLM_BASE_URL(기본 https://api.openai.com/v1) · LLM_API_KEY · "
                   f"LLM_MODEL(기본 {API_DEFAULT_MODEL}). 키는 .env로만 주입하며 이 화면은 키를 입력받지 않습니다. "
                   "비용은 README '예상 비용' 절을 참조하세요.")
        st.caption("LLM_API_KEY: " + ("설정됨 또는 .env 있음" if has_key else "미설정 - 실행 시 fail-closed로 실패합니다"))

    clicked = st.button("리포트 생성", type="primary", use_container_width=True)
    return clicked, mode, opts


def _rule_counts(stats: dict) -> dict[str, int]:
    counts = {r: 0 for r in SEED_RULES}
    for t in stats["tasks"].values():
        for dist in t["violations_by_attempt"].values():
            for rule, n in dist.items():
                counts[rule] = counts.get(rule, 0) + n
    return counts


def _run_panel(run: dict | None, stale: bool) -> None:
    _section("out", "이번 실행")
    if run is None:
        st.caption("아직 실행 전입니다. 왼쪽에서 값을 바꾸고 '리포트 생성'을 누르면 위기 게이트 → 생성 → "
                   "가드레일 순서로 실행되고, 위 리포트의 '생성 대기' 자리가 실제 문장으로 바뀝니다.")
        return
    if stale:
        st.info("입력이 바뀌어 이전 실행 결과를 내렸습니다. 위 리포트는 결정론 부분만 갱신된 미리보기입니다. "
                "'리포트 생성'을 다시 누르세요.")
        return
    if run["crisis"]:
        st.error("위기 신호 검출 → LLM 호출 0회. 해설 대신 상담 연결 안내만 생성했습니다 (입력 게이트, fail-closed).")
        st.caption("검출 패턴 (평가자 확인용 - 보호자 화면에는 표시되지 않음): " + ", ".join(run["crisis"]))
        return

    stats = run["stats"]
    if run["mode"] == "mock":
        st.caption("생성 모드: mock 템플릿 - 실제 LLM 생성이 아님. 가드레일·품질 지표는 실제 규칙이 실제로 잰 값입니다.")
    else:
        st.caption(f"생성 모드: {MODES[run['mode']]} · 모델 {stats['model']}")

    regen = sum(t["regen_count"] for t in stats["tasks"].values())
    fallback = [f"{task}.{block}" for task, t in stats["tasks"].items()
                for block, state in t["block_states"].items() if state == "fallback"]
    tok_in, tok_out = stats["total_prompt_tokens"], stats["total_completion_tokens"]
    c = st.columns(5)
    c[0].metric("재생성 호출", f"{regen}회")
    c[1].metric("폴백 블록", f"{len(fallback)}개")
    c[2].metric("LLM 호출", f"{len(stats['llm_calls'])}회")
    c[3].metric("토큰 in / out", f"{tok_in} / {tok_out}" if (tok_in or tok_out) else "- (mock)")
    c[4].metric("LLM 시간", f"{stats['total_llm_seconds']}s")
    if fallback:
        st.caption("안전 문구로 대체된 블록: " + ", ".join(fallback))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**걸린 규칙 분포 (G1~G10, 전 시도 합산)**")
        counts = _rule_counts(stats)
        if any(counts.values()):
            st.bar_chart(pd.DataFrame({"위반 수": [counts[r] for r in SEED_RULES]}, index=list(SEED_RULES)),
                         height=220)
        else:
            st.caption("위반 0건 - 모든 블록이 첫 시도에서 통과했습니다.")
    with right:
        st.markdown("**블록 상태 (5개)**")
        st.table(pd.DataFrame([{"task": task, "block": block, "state": state}
                               for task, t in stats["tasks"].items()
                               for block, state in t["block_states"].items()]))

    q = stats["quality"]
    j, r, w = q["jargon"], q["reflection"], q["direction_warnings"]
    _section("out", "품질 지표", "측정만, 게이트 아님")
    qc = st.columns(3)
    qc[0].metric("보호자 표현 반영률 (항목 · 토큰)",
                 f"{r['items_reflected']}/{r['items_total']} ({fmt_rate(r['item_rate'])})",
                 help=f"토큰 {len(r['tokens_hit'])}/{len(r['tokens'])} ({fmt_rate(r['token_rate'])})")
    qc[1].metric("용어 잔존 (등장 · 용어 블록 · 풀이 동반)",
                 f"{j['term_hits']}회 · {j['blocks_with_term']}/{j['blocks_total']} · {fmt_rate(j['gloss_rate'])}")
    qc[2].metric("질문 방향 경고", f"{len(w)}건")
    with st.expander("위반 상세 (시도별) · 표현 반영 토큰 · 방향 경고"):
        if run["violations"]:
            st.dataframe(pd.DataFrame(run["violations"]), use_container_width=True, hide_index=True)
        else:
            st.caption("위반 없음")
        st.caption("반영된 보호자 토큰: " + (", ".join(r["tokens_hit"]) or "-")
                   + " / 추출 토큰: " + (", ".join(r["tokens"]) or "-"))
        for warn in w:
            st.caption(f"WARN {warn['block']}: '{warn['matched']}' - {warn['question']}")


# ---------------------------------------------------------------- 메인

st.set_page_config(page_title="CBCL 동반 가이드 · 탐색 콘솔", layout="wide")
st.markdown(PANEL_CSS, unsafe_allow_html=True)
_init_state()
st.title("CBCL 동반 가이드 · 평가자용 탐색 콘솔")
st.caption("슬라이더로 프로파일을 바꾸면 곡선·밴드·고정 문구가 즉시 갱신됩니다 (LLM 미호출). "
           "'리포트 생성'을 누르면 위기 게이트 → 생성 → 가드레일을 거쳐 생성 문구가 채워지고, 아래 패널에 "
           "재생성·폴백·규칙 분포·품질 지표가 나옵니다. 모든 프로파일은 자작 가상 데이터입니다.")
st.markdown(f'<div class="cb-guide">{_badge("in", 1)}<span>왼쪽 입력 패널에서 프로파일과 생성 설정을 바꾸고,</span>'
            f'{_badge("out", 2)}<span>오른쪽 결과 패널에서 리포트와 실행 지표를 확인합니다.</span></div>',
            unsafe_allow_html=True)

left_col, right_col = st.columns([2, 3], gap="large")
input_box = left_col.container(border=True, key="input_panel")
result_box = right_col.container(border=True, key="result_panel")
with input_box:
    _panel_head("in", 1, "입력", "프로파일과 생성 설정")
    clicked, mode, opts = _left_panel()
with result_box:
    _panel_head("out", 2, "결과", "보호자 리포트와 실행 패널")

inputs = _collect_inputs()
raw = build_profile_raw(inputs)
fingerprint = json.dumps(raw, sort_keys=True, ensure_ascii=False)
try:
    profile = build_profile(inputs)
except ProfileError as e:
    with result_box:
        st.error("입력 프로파일 검증 실패 - 리포트를 생성하지 않습니다 (파서 fail-closed)")
        for msg in e.errors:
            st.write(f"- {msg}")
    st.stop()

autorun = st.session_state.pop("autorun", False)
if clicked or autorun:
    with result_box, st.spinner("위기 게이트 → 생성 → 가드레일 실행 중..."):
        try:
            st.session_state["run"] = _run_pipeline(profile, mode, opts, fingerprint)
        except llm_failure_types() as e:
            st.session_state.pop("run", None)
            st.error(f"LLM 호출 실패: {e} - 생성이 완료되지 않아 리포트를 교체하지 않습니다 (fail-closed).")

run = st.session_state.get("run")
stale = bool(run) and run["fingerprint"] != fingerprint
with result_box:
    _section("out", "리포트", "2페이지")
    if run and not stale:
        st.caption("생성 완료 - 위기 안내 화면이거나, 가드레일을 통과한 생성 문구로 채워진 리포트입니다.")
        html = run["html"]
    else:
        st.caption("생성 전 미리보기 - 곡선·밴드·구간·고정 문구는 실제 값이고, LLM 생성 자리(연결 문단·질문·관찰·요약)는 "
                   "'생성 대기'입니다. 같은 템플릿과 렌더러입니다.")
        html = build_pending_report_html(profile)
    components.html(html, height=REPORT_HEIGHT, scrolling=True)
    _run_panel(run, stale)
