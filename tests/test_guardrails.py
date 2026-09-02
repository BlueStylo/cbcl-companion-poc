"""가드레일 결정론 테스트 (LLM 호출 없음).

기준 출력은 mock fixture의 클린 응답을 쓰고, 규칙별로 위반을 주입해
검출되는지 확인한다.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.guardrails import check_output, run_with_guardrails
from src.parser import load_profile

PROFILE = load_profile(ROOT / "data/profiles/p2_partial_borderline.json")
FIXTURE = json.loads((ROOT / "data/fixtures/p2_partial_borderline.json").read_text(encoding="utf-8"))


@pytest.fixture()
def explain_out():
    return copy.deepcopy(FIXTURE["explain"]["attempts"][0])


@pytest.fixture()
def prep_out():
    return copy.deepcopy(FIXTURE["prep"]["attempts"][0])


def rules(violations):
    return {v.rule_id for v in violations}


def test_clean_outputs_pass(explain_out, prep_out):
    assert check_output(PROFILE, "explain", explain_out) == []
    assert check_output(PROFILE, "prep", prep_out) == []


def test_g1_diagnosis_detected_even_in_negation(explain_out):
    """R1은 예외 없음: 'ADHD가 아닙니다'도 차단 (의도된 과검출)."""
    explain_out["overview"] = "이 결과만으로 보면 ADHD가 아닙니다."
    assert "G1" in rules(check_output(PROFILE, "explain", explain_out))


def test_g2_severity_both_directions(explain_out):
    explain_out["before_counseling"] = "이 점수는 심각한 수준입니다."
    assert "G2" in rules(check_output(PROFILE, "explain", explain_out))
    explain_out["before_counseling"] = "걱정하지 않으셔도 됩니다. 괜찮습니다."
    assert "G2" in rules(check_output(PROFILE, "explain", explain_out))


def test_g3_fabricated_number_in_text(explain_out):
    explain_out["overview"] += " 백분위로는 상위 3%입니다."
    assert "G3" in rules(check_output(PROFILE, "explain", explain_out))


def test_g3_number_in_prep_observation(prep_out):
    prep_out["observation_points"][0]["point"] = "T점수 75를 넘는 날이 있는지 기록해 두기"
    assert "G3" in rules(check_output(PROFILE, "prep", prep_out))


def test_g3_allowed_numbers_pass(explain_out):
    """입력 수치와 T점수 체계 상수(평균 50 등)는 위조가 아니다."""
    explain_out["before_counseling"] = "T점수는 또래 평균을 50으로 놓고 비교한 점수이며, 주의집중 척도는 T점수 67입니다."
    assert "G3" not in rules(check_output(PROFILE, "explain", explain_out))


def test_g6_prescription_detected(explain_out):
    """처방·치료 권고: 약물, 치료 시작, 의료기관 방문 지시 모두 차단."""
    explain_out["before_counseling"] = "필요하면 약물 치료를 시작해 보세요."
    assert "G6" in rules(check_output(PROFILE, "explain", explain_out))
    explain_out["before_counseling"] = "가까운 병원에 방문해 진료를 받아 보세요."
    assert "G6" in rules(check_output(PROFILE, "explain", explain_out))


def test_g6_allowed_counsel_guidance_passes(explain_out):
    """허용된 유일한 형태('예약된 상담에서...')는 G6에 걸리지 않는다."""
    explain_out["before_counseling"] = "궁금한 점은 예약된 상담에서 상담사와 이야기해 보세요."
    assert "G6" not in rules(check_output(PROFILE, "explain", explain_out))


def test_g4_invalid_source_scale(prep_out):
    prep_out["questions_for_counselor"][0]["source_scale"] = "focus_ability"
    assert "G4" in rules(check_output(PROFILE, "prep", prep_out))


def test_g5_question_count(prep_out):
    prep_out["questions_for_counselor"] = prep_out["questions_for_counselor"][:2]
    assert "G5" in rules(check_output(PROFILE, "prep", prep_out))


def test_fallback_flag_cannot_bypass_checks(explain_out):
    """LLM 출력이 _fallback을 흉내 내도 검사를 우회하지 못한다."""
    explain_out["overview"] = "ADHD로 보입니다."

    def gen_fn(attempt, pending, feedback):
        out = copy.deepcopy(explain_out)
        out["_fallback"] = True
        return out

    result = run_with_guardrails(PROFILE, "explain", gen_fn)
    assert "overview" in result.fallback_blocks
    assert check_output(PROFILE, "explain", result.output) == []


def test_fail_closed_fallback_after_two_regens(explain_out):
    """계속 위반하는 블록은 재생성 2회 후 안전 문구로 대체되고,

    최종 출력에는 잔존 위반이 없어야 한다 (fail-closed).
    """
    calls = []

    def gen_fn(attempt, pending, feedback):
        calls.append(attempt)
        out = copy.deepcopy(explain_out)
        out["before_counseling"] = "이 정도면 아무 문제 없습니다."
        return out

    result = run_with_guardrails(PROFILE, "explain", gen_fn)
    assert calls == [0, 1, 2]
    assert result.regen_count == 2
    assert result.fallback_blocks == ["before_counseling"]
    assert check_output(PROFILE, "explain", result.output) == []


# ---------------------------------------------------------------- G7/G8/G9

def test_g7_scale_id_leak_in_question(prep_out):
    """실LLM 실측 결함 (b): 질문 본문에 '(scale_id: 'attention')' 누출."""
    prep_out["questions_for_counselor"][0]["question"] += " (scale_id: 'attention')"
    assert "G7" in rules(check_output(PROFILE, "prep", prep_out))


def test_g7_lowercase_identifier_in_overview(explain_out):
    explain_out["overview"] = "위축(withdrawn) 척도의 T점수 62는 준임상 범위입니다."
    assert "G7" in rules(check_output(PROFILE, "explain", explain_out))


def test_g7_uppercase_abbreviations_are_not_leaks(explain_out):
    """대문자 약어(T점수, K-CBCL, SEM)는 형식 누출이 아니다."""
    explain_out["before_counseling"] = "K-CBCL 6-18은 T점수로 표기되며 SEM은 측정 오차입니다. 결과의 해석은 예약된 상담에서 상담사와 이야기해 보세요."
    assert "G7" not in rules(check_output(PROFILE, "explain", explain_out))


def test_g7_not_applied_to_counselor_briefing(prep_out):
    """상담사용 사전 요약은 보호자에게 노출되지 않으므로 G7 대상이 아니다."""
    prep_out["counselor_briefing"] += " 관련 척도: attention, withdrawn."
    assert "G7" not in rules(check_output(PROFILE, "prep", prep_out))


def test_g8_nonstandard_band_vocab(explain_out):
    """실LLM 실측 결함 (d): 총 문제행동 정상을 '경계 수준'으로 서술."""
    explain_out["overview"] = "총 문제행동과 내재화 문제는 경계 수준으로 나타났고, 외현화 문제는 정상 범위에 속합니다."
    assert "G8" in rules(check_output(PROFILE, "explain", explain_out))
    for bad in ("경계성(borderline) 영역", "위험군에 해당", "다소 높은 편", "준임계 범위"):
        explain_out["overview"] = f"내재화 문제는 {bad}입니다."
        assert "G8" in rules(check_output(PROFILE, "explain", explain_out)), bad


def test_g8_label_mismatch_per_mentioned_scale(explain_out):
    """여러 척도를 언급하는 블록은 언급 척도별로 대조 (총 문제행동 정상 → 준임상 오기)."""
    explain_out["overview"] = "총 문제행동 T점수 57(준임상), 내재화 문제 T점수 62(준임상), 외현화 문제 T점수 52(정상)으로 보고되었습니다."
    vs = [v for v in check_output(PROFILE, "explain", explain_out) if v.rule_id == "G8"]
    assert len(vs) == 1 and "총 문제행동" in vs[0].matched


def test_g8_uses_item_scale_when_no_scale_mentioned(prep_out):
    """척도 언급이 없으면 항목의 source_scale과 대조한다."""
    prep_out["questions_for_counselor"][0]["question"] = "이 결과가 정상 범위라면 무엇을 더 살펴보면 될까요?"  # source attention(준임상)
    assert "G8" in rules(check_output(PROFILE, "prep", prep_out))


def test_g8_skips_when_no_scale_attributable(explain_out):
    """귀속할 척도가 없는 밴드 어휘는 대조하지 않는다 (규칙 한계, README 명시)."""
    explain_out["before_counseling"] = "정상 범위라는 말을 들어도 마음이 놓이지 않는 것은 자연스럽습니다."
    assert "G8" not in rules(check_output(PROFILE, "explain", explain_out))


def test_g8_general_usage_of_words_is_not_a_label(explain_out, prep_out):
    """'임상적', '비정상', '정상적', '경계를' 같은 일반어 용법은 라벨이 아니다."""
    explain_out["before_counseling"] = "임상적 해석은 상담사가 합니다. 관찰자마다 결과가 다른 것이 비정상은 아니며, 예약된 상담에서 상담사와 이야기해 보세요."
    assert "G8" not in rules(check_output(PROFILE, "explain", explain_out))
    prep_out["questions_for_counselor"][0]["question"] = "이 검사로 알 수 있는 것과 없는 것의 경계를 어디에 두면 될까요?"
    assert "G8" not in rules(check_output(PROFILE, "prep", prep_out))


def test_g9_normal_scale_cannot_anchor_question_or_observation(prep_out):
    """실LLM 실측 결함 (c): 정상 범위 척도(신체증상 T=51)에 대해 증상이 있는 것처럼 묻는 질문."""
    prep_out["questions_for_counselor"][0] = {
        "question": "신체 증상이 언제, 어떤 상황에서 나타나는지 어떻게 살펴보면 될까요?",
        "source_scale": "somatic"}
    assert "G9" in rules(check_output(PROFILE, "prep", prep_out))
    prep_out = copy.deepcopy(FIXTURE["prep"]["attempts"][0])
    prep_out["observation_points"][0]["source_scale"] = "total_problems"  # 종합지표도 포함
    assert "G9" in rules(check_output(PROFILE, "prep", prep_out))


def test_g9_skipped_when_all_scales_normal():
    """전 척도 정상 프로파일(P1)은 앵커가 없으므로 G9를 적용하지 않는다."""
    profile = load_profile(ROOT / "data/profiles/p1_all_normal.json")
    out = json.loads((ROOT / "data/fixtures/p1_all_normal.json").read_text(encoding="utf-8"))["prep"]["attempts"][0]
    assert all(it["source_scale"] in {"total_problems", "anxious_depressed", "social_immaturity"}
               for it in out["questions_for_counselor"] + out["observation_points"])
    assert "G9" not in rules(check_output(profile, "prep", out))


def test_g2_allows_emotion_acknowledgement_but_blocks_verdict(explain_out):
    """감정 인정 허용선: 보호자의 감정을 인정하는 문장은 통과, 결과 판정은 차단."""
    for ok in ("결과를 보고 걱정되는 마음이 드는 것은 자연스럽습니다.",
               "숫자를 보면 마음이 무거워질 수 있습니다. 궁금한 점은 예약된 상담에서 상담사와 이야기해 보세요.",
               "궁금하고 불안한 마음이 드는 것은 당연합니다."):
        explain_out["before_counseling"] = ok
        assert "G2" not in rules(check_output(PROFILE, "explain", explain_out)), ok
    for bad in ("이 정도면 괜찮습니다.", "걱정하지 않으셔도 됩니다.", "안심하셔도 됩니다.",
                "전반적으로 큰 문제는 없어 보입니다.",    # 새 구조 실측에서 통과했던 결과 판정
                "전반적으로 안정적인 상태를 유지하고 있는 것으로 보입니다."):
        explain_out["before_counseling"] = bad
        assert "G2" in rules(check_output(PROFILE, "explain", explain_out)), bad


def test_new_rules_regenerate_then_fallback(prep_out):
    """G7/G8/G9도 기존 규칙과 같은 루프: 블록 재생성 2회 후 안전 문구 폴백."""
    prep_out["questions_for_counselor"][0]["question"] += " (scale_id: 'attention')"
    calls = []

    def gen_fn(attempt, pending, feedback):
        calls.append(attempt)
        if attempt > 0:
            assert any(v.rule_id == "G7" for v in feedback)
        return copy.deepcopy(prep_out)

    result = run_with_guardrails(PROFILE, "prep", gen_fn)
    assert calls == [0, 1, 2]
    assert result.fallback_blocks == ["questions_for_counselor"]
    assert check_output(PROFILE, "prep", result.output) == []


def test_explain_schema_is_overview_and_before_counseling_only(explain_out):
    """LLM 범위 축소: explain 출력은 연결 문단 + 상담 전 안내 두 블록뿐이다."""
    assert set(explain_out) == {"overview", "before_counseling"}
    extra = dict(explain_out, scale_explanations=[{"scale_id": "attention", "what_the_number_means": "심각합니다."}])
    assert check_output(PROFILE, "explain", extra) == []   # 스키마 밖 필드는 렌더링되지 않으므로 검사 대상도 아니다
    del explain_out["before_counseling"]
    assert "G5" in rules(check_output(PROFILE, "explain", explain_out))


def test_g10_example_phrase_only_allowed_when_caregiver_wrote_it(prep_out):
    """예시 오염: 프롬프트 예시의 관찰(학원 숙제)은 그 말을 쓴 보호자(p2)에게만 인용할 수 있다."""
    assert "G10" not in rules(check_output(PROFILE, "prep", prep_out))   # p2 의견에 '학원 숙제'가 있다
    p5a = load_profile(ROOT / "data/profiles/p5a_paired_notes.json")
    out = json.loads((ROOT / "data/fixtures/p5a_paired_notes.json").read_text(encoding="utf-8"))["prep"]["attempts"][0]
    assert "G10" not in rules(check_output(p5a, "prep", out))
    out["questions_for_counselor"][0]["question"] = "학원 숙제 앞에서 딴 곳을 자주 보는 모습은 주의집중 척도(T점수 67)와 관련이 있을까요?"
    assert "G10" in rules(check_output(p5a, "prep", out))
