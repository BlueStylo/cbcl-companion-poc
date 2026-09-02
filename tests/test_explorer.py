"""탐색 콘솔 테스트 (Streamlit 미사용, LLM 호출 없음).

슬라이더 값 → 프로파일 조립이 파서를 통과하는지, 종합 지표 참고 힌트 로직,
템플릿 목이 가드레일을 통과하는지(잔존 위반 0)와 시드·자기 교정 동작,
생성 전 미리보기 렌더 경로를 검사한다.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.profile_builder import (EXAMPLE_ORDER, ExplorerInputs, build_profile, build_profile_raw,
                                 composite_hints, example_inputs)
from src.generator import generate_all
from src.guardrails import check_output
from src.llm_client import TemplateMockClient
from src.parser import COMPOSITE_IDS, SYNDROME_IDS, ProfileError, load_profile, parse_profile
from src.quality import reflection_metrics
from src.report_html import build_pending_report_html, build_preview_html

ALL_IDS = (*COMPOSITE_IDS, *SYNDROME_IDS)


def _profile(pid):
    return load_profile(ROOT / f"data/profiles/{pid}.json")


@pytest.mark.parametrize("pid", EXAMPLE_ORDER)
def test_example_values_roundtrip_through_parser(pid):
    """예시 8종 → 위젯 값 → 조립 → 파서: 수치·밴드·의견이 원본과 같다."""
    original = _profile(pid)
    rebuilt = build_profile(example_inputs(pid))
    assert {s.scale_id: (s.t_score, s.band) for s in rebuilt.all_scales()} \
        == {s.scale_id: (s.t_score, s.band) for s in original.all_scales()}
    assert rebuilt.caregiver_notes == original.caregiver_notes
    assert rebuilt.days_until_counseling == original.days_until_counseling


def test_slider_values_get_bands_from_parser_rules_and_forged_band_is_rejected():
    """밴드 라벨은 콘솔이 아니라 파서 규칙(개별 60/70, 종합 60/63)이 정한다."""
    ts = {sid: 50 for sid in ALL_IDS}
    ts.update(attention=60, aggressive=70, internalizing=60, total_problems=63)
    profile = build_profile(ExplorerInputs(t_scores=ts, notes=["메모 한 줄"]))
    bands = {s.scale_id: s.band for s in profile.all_scales()}
    assert bands["attention"] == "borderline" and bands["aggressive"] == "clinical"
    assert bands["internalizing"] == "borderline" and bands["total_problems"] == "clinical"
    assert bands["withdrawn"] == "normal" and bands["externalizing"] == "normal"
    assert profile.profile_type == "clinical"
    # 조립 결과의 라벨을 손으로 바꿔 넣으면 파서가 거부한다 (규칙 한 벌)
    raw = build_profile_raw(ExplorerInputs(t_scores=ts))
    raw["syndromes"][0]["band"] = "clinical"
    with pytest.raises(ProfileError):
        parse_profile(raw)


def test_composite_hint_only_when_all_members_elevated_and_composite_normal():
    ts = {sid: 50 for sid in ALL_IDS}
    assert composite_hints(ts) == []
    ts.update(withdrawn=61, somatic=60, anxious_depressed=62, internalizing=55)
    hints = composite_hints(ts)
    assert len(hints) == 1 and "내재화 문제" in hints[0]
    ts["internalizing"] = 60                     # 종합 지표가 준임상이면 힌트 없음
    assert composite_hints(ts) == []
    ts.update(internalizing=55, somatic=50)      # 하위 척도 하나라도 정상이면 힌트 없음
    assert composite_hints(ts) == []
    ts.update(delinquent=60, aggressive=70, externalizing=50)
    assert any("외현화 문제" in h for h in composite_hints(ts))


@pytest.mark.parametrize("pid", ["p2_partial_borderline", "p4_clinical", "p5a_paired_notes"])
def test_template_mock_passes_guardrails_first_try(pid):
    """템플릿 목: 첫 시도 위반 0, 재생성 0, 폴백 0, 최종 잔존 위반 0. 근거는 준임상 이상 척도만."""
    profile = _profile(pid)
    client = TemplateMockClient()
    assert check_output(profile, "prep", client.generate("prep", profile, 0, "", "")) == []
    results = generate_all(profile, client)
    assert list(results) == ["prep"]
    for task, r in results.items():
        assert r.regen_count == 0 and r.fallback_blocks == []
        assert check_output(profile, task, r.output) == []
    prep = results["prep"].output
    elevated = {s.scale_id for s in profile.elevated_scales()}
    assert set(prep) == {"questions_for_counselor"}                # 관찰 포인트는 목이 만들지 않는다 (결정론 조립)
    assert {it["source_scale"] for it in prep["questions_for_counselor"]} <= elevated
    assert 5 <= len(prep["questions_for_counselor"]) <= 7
    assert reflection_metrics(profile, prep)["item_rate"] >= 0.5   # 보호자 문장을 인용한다


def test_template_mock_seed_is_caught_and_self_correction_drops_offending_quotes():
    p2 = _profile("p2_partial_borderline")
    # 시드: 첫 시도에서 검출 → 재생성 1회로 회복 (폴백 0)
    results = generate_all(p2, TemplateMockClient(seed_rules={"G1", "G7"}))
    first = {v.rule_id for r in results.values() for v in r.violations if v.attempt == 0}
    assert {"G1", "G7"} <= first
    assert results["prep"].regen_count == 1 and results["prep"].fallback_blocks == []
    # 새 규칙 시드(G3 숫자, G10 예시 오염, G11 방향)도 첫 시도에서 검출되고 재생성 1회로 회복된다
    results = generate_all(p2, TemplateMockClient(seed_rules={"G3", "G11"}))
    first = {v.rule_id for r in results.values() for v in r.violations if v.attempt == 0}
    assert {"G3", "G11"} <= first and results["prep"].regen_count == 1
    results = generate_all(_profile("p5a_paired_notes"), TemplateMockClient(seed_rules={"G10"}))
    assert "G10" in {v.rule_id for r in results.values() for v in r.violations if v.attempt == 0}
    # 시드 지속: 재생성 2회 소진 → 안전 문구 폴백, 최종 출력은 깨끗하다 (fail-closed)
    persisted = generate_all(p2, TemplateMockClient(seed_rules={"G1"}, persist_seed=True))
    assert persisted["prep"].regen_count == 2 and persisted["prep"].fallback_blocks == ["questions_for_counselor"]
    assert check_output(p2, "prep", persisted["prep"].output) == []
    # A1: 보호자 의견 자체에 진단명·판정 요구 → 첫 시도 G1/G2, 재생성에서 그 인용을 빼고 통과
    a1 = _profile("a1_adversarial")
    results = generate_all(a1, TemplateMockClient())
    first = {v.rule_id for r in results.values() for v in r.violations if v.attempt == 0}
    assert {"G1", "G2"} <= first
    assert all(r.regen_count == 1 and r.fallback_blocks == [] for r in results.values())
    assert "ADHD" not in json.dumps({t: r.output for t, r in results.items()}, ensure_ascii=False)
    # P1 전 척도 정상: 총 문제행동 근거의 예외 경로도 첫 시도 통과
    results = generate_all(_profile("p1_all_normal"), TemplateMockClient())
    assert all(r.regen_count == 0 and r.fallback_blocks == [] for r in results.values())


def test_template_mock_output_has_no_digits_and_quotes_long_notes_partially():
    """템플릿 목은 새 계약을 지킨다: 숫자 없음, 질문 25~90자, 긴 의견은 앞부분만 「」 인용해도 G10 (a)를 만족."""
    import re
    long_note = "학원 숙제를 앞에 두면 딴 데를 자주 보고, 저녁마다 숙제를 미루다가 밤늦게야 겨우 시작하는 날이 많아졌습니다"
    profile = _profile("p2_partial_borderline").model_copy(update={"caregiver_notes": [long_note]})
    out = TemplateMockClient().generate("prep", profile, 0, "", "")
    assert check_output(profile, "prep", out) == []
    texts = [q["question"] for q in out["questions_for_counselor"]]
    assert not any(re.search(r"\d", t) for t in texts)
    assert all(25 <= len(q["question"]) <= 90 for q in out["questions_for_counselor"])
    assert any("…」" in t for t in texts)


def test_pending_report_renders_deterministic_parts_without_llm():
    """생성 전 미리보기: 곡선·오차 범위선·수치·관찰 포인트는 실제 값, 질문 자리는 '생성 대기', 검증 통과 배지는 없다."""
    html = build_pending_report_html(_profile("p2_partial_borderline"))
    assert "생성 대기" in html and "검증 통과" not in html and "안전 문구" not in html
    assert "이번 검사 결과 67T" in html and "<svg" in html and "준임상" in html
    questions = html[html.index('id="question-list"'):html.index("</ul>", html.index('id="question-list"'))]
    assert "근거:" not in questions                                  # 자리표시 항목에는 근거 배지를 붙이지 않는다
    assert html.count("근거:") == 3                                  # 결정론 관찰 포인트 3개에는 붙는다


def test_preview_gate_shows_crisis_screen_before_generation():
    """미리보기 게이트: 보호자 의견에 위기 표현이 있으면 생성 버튼 전에도 점수 리포트 대신 위기 안내를 돌려준다.

    입력 게이트와 같은 사전(detect_crisis_signals)을 쓰고, 의견을 고치면 다시 점수 미리보기로 돌아온다. LLM은 없다.
    """
    p2 = _profile("p2_partial_borderline")
    html, crisis = build_preview_html(p2)
    assert crisis == [] and "생성 대기" in html and "이번 검사 결과 67T" in html
    flagged = p2.model_copy(update={"caregiver_notes": ["아빠가 때려요", p2.caregiver_notes[1]]})
    html, crisis = build_preview_html(flagged)
    assert crisis and "상담 연결 안내" in html and "1577-0199" in html
    assert "T점수" not in html and "67T" not in html and "생성 대기" not in html    # 점수와 곡선을 먼저 보여 주지 않는다
    assert "아빠가 때려요" not in html                                              # 검출 원문도 재노출하지 않는다
    html, crisis = build_preview_html(_profile("c1_crisis"))
    assert crisis and "상담 연결 안내" in html
