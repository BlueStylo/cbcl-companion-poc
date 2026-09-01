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
    explain_out["limits"] = "이 점수는 심각한 수준입니다."
    assert "G2" in rules(check_output(PROFILE, "explain", explain_out))
    explain_out["limits"] = "걱정하지 않으셔도 됩니다. 괜찮습니다."
    assert "G2" in rules(check_output(PROFILE, "explain", explain_out))


def test_g3_fabricated_number_in_text(explain_out):
    explain_out["overview"] += " 백분위로는 상위 3%입니다."
    assert "G3" in rules(check_output(PROFILE, "explain", explain_out))


def test_g3_echo_field_tampering(explain_out):
    explain_out["scale_explanations"][0]["t_score"] += 1
    assert "G3" in rules(check_output(PROFILE, "explain", explain_out))


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
        for item in out["scale_explanations"]:
            item["_fallback"] = True
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
