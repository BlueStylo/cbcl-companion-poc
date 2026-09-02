"""가드레일 결정론 테스트 (LLM 호출 없음).

기준 출력은 mock fixture의 클린 응답(prep, 질문 1블록)을 쓰고, 규칙별로 위반을
주입해 검출되는지, 정당한 문장은 통과하는지(양성·음성)를 확인한다. LLM 블록은 질문뿐이다
(ADR 0010과 그 보강): 연결 문단, 관찰 포인트, 상담사 요약은 결정론 조립이라 여기서 다루지 않는다.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import PROMPTS_DIR, TASKS, build_user_message, load_system_prompt
from src.guardrails import (SAFE_QUESTION, TASK_BLOCKS, Violation, _check_crisis_vocab,
                            _strip_fallback_flags, check_output, detect_crisis_signals,
                            has_quote_claim, quotes_caregiver_note, run_with_guardrails,
                            scale_mentions)
from src.llm_client import TemplateMockClient
from src.parser import load_profile

PROFILE = load_profile(ROOT / "data/profiles/p2_partial_borderline.json")
FIXTURE = json.loads((ROOT / "data/fixtures/p2_partial_borderline.json").read_text(encoding="utf-8"))


@pytest.fixture()
def prep_out():
    return copy.deepcopy(FIXTURE["prep"]["attempts"][0])


def rules(violations):
    return {v.rule_id for v in violations}


def _q(prep_out, i, text, source=None):
    prep_out["questions_for_counselor"][i]["question"] = text
    if source:
        prep_out["questions_for_counselor"][i]["source_scale"] = source
    return prep_out


# ---------------------------------------------------------------- 구조

def test_llm_schema_is_prep_with_question_block_only(prep_out):
    """LLM 태스크는 prep 하나, 블록은 질문 1개 (ADR 0010과 그 보강). 옛 explain 태스크와 observation_points 블록은 없다."""
    assert TASK_BLOCKS == {"prep": ("questions_for_counselor",)}
    assert TASKS == ("prep",)
    assert set(prep_out) == {"questions_for_counselor"}
    assert check_output(PROFILE, "prep", prep_out) == []
    with pytest.raises(KeyError):
        load_system_prompt("explain")
    assert not (PROMPTS_DIR / "explainer_system.md").exists()


@pytest.mark.parametrize("pid", ["p1_all_normal", "p2_partial_borderline", "p3_boundary_mix", "p4_clinical",
                                 "p5a_paired_notes", "p5b_paired_notes"])
def test_all_clean_fixtures_pass(pid):
    profile = load_profile(ROOT / f"data/profiles/{pid}.json")
    out = json.loads((ROOT / f"data/fixtures/{pid}.json").read_text(encoding="utf-8"))["prep"]["attempts"][0]
    assert check_output(profile, "prep", out) == []


def test_legacy_keys_are_ignored_and_dropped(prep_out):
    """스키마 밖 키 정책: 옛 스키마의 overview·counselor_briefing·observation_points가 와도 위반이 아니라
    무시되며, rebuild가 버려 리포트에 닿지 않는다. G5는 필수 키 누락만 잡는다."""
    legacy = dict(prep_out, overview="ADHD로 보입니다. 병원에 가 보세요.",
                  counselor_briefing="총 문제행동 T=57(정상). 약물 치료 권고.",
                  observation_points=[{"point": "약물 치료 후보를 적어 두기", "source_scale": "somatic"}])
    assert check_output(PROFILE, "prep", legacy) == []
    result = run_with_guardrails(PROFILE, "prep", lambda a, p, f: copy.deepcopy(legacy))
    assert set(result.output) == {"questions_for_counselor"}
    assert result.violations == [] and result.fallback_blocks == [] and result.block_count == 1
    del legacy["questions_for_counselor"]
    assert "G5" in rules(check_output(PROFILE, "prep", legacy))


def test_prompt_has_no_trace_of_removed_blocks():
    """프롬프트 계약에 explain·연결 문단·상담사 요약·상담 전 안내·관찰 포인트 키가 없고, 출력 키는 1개뿐이다."""
    system = load_system_prompt("prep")
    for absent in ("overview", "counselor_briefing", "before_counseling", "observation_points", "explainer", "explain", "마음가짐"):
        assert absent not in system, absent
    assert '"questions_for_counselor"' in system and system.count('"question"') >= 5
    assert "위기 표현을 쓰지 않습니다" in system                     # G12 계약
    assert "절대 규칙" in system and "권장 사항" in system
    # 예시 질문에 숫자가 없다 (G3 계약과 일치)
    import re
    example = system[system.index("작성 예시"):]
    assert not re.search(r'"question": "[^"]*\d', example)
    first = build_user_message(PROFILE, 0, ["questions_for_counselor"], [])
    regen = build_user_message(PROFILE, 1, ["questions_for_counselor"],
                               [Violation("G3", "questions_for_counselor", "67", 0)])
    assert "caregiver_notes" in first and "블록 questions_for_counselor" in regen
    assert "아라비아 숫자" in regen                     # RULE_HINTS의 새 G3 정의가 피드백에 실린다


# ---------------------------------------------------------------- G1 / G2 / G6

def test_g1_diagnosis_detected_even_in_negation(prep_out):
    """R1은 예외 없음: 'ADHD가 아닙니다'도 차단 (의도된 과검출). Codex 점검의 '양극성 장애'도 잡는다."""
    assert "G1" in rules(check_output(PROFILE, "prep", _q(prep_out, 0, "이 결과만으로 보면 ADHD가 아니라고 봐도 되나요?")))
    assert "G1" in rules(check_output(PROFILE, "prep", _q(prep_out, 0, "양극성 장애가 의심되는 기분 변화는 상담에서 어떻게 살펴보게 되나요?")))


def test_g2_severity_both_directions(prep_out):
    for bad in ("주의집중 척도 결과가 심각한 편이라 바로 조치가 필요한 상황인가요?",
                "현재 상태가 매우 위중한지 상담에서 들을 수 있을까요?",
                "이 정도 결과라면 크게 걱정하지 않으셔도 되는 수준인가요?",
                "전반적으로 큰 문제는 없어 보이는 결과라고 읽어도 되나요?"):
        assert "G2" in rules(check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 0, bad))), bad
    ok = "결과를 보고 걱정되는 마음이 드는 것은 자연스러운지 상담에서 이야기해 봐도 될까요?"
    assert "G2" not in rules(check_output(PROFILE, "prep", _q(prep_out, 0, ok)))


def test_g6_prescription_detected_and_counsel_guidance_allowed(prep_out):
    for bad in ("지금이라도 상담과 별개로 치료를 시작하는 것이 좋을까요?",
                "정확한 확인을 위해 가까운 병원을 방문해야 하는 단계인가요?",
                "놀이치료를 고려해 보는 것이 좋은지 상담에서 물어봐도 되나요?",
                "전문의 상담이 필요한 수준인지 이 결과만으로 알 수 있나요?",
                # 조사 변형과 완곡한 의료기관 안내: '치료를 고려'만 막던 빈틈
                "학원 숙제 앞에서 딴 데를 보는 정도면 치료도 고려해야 하는 수준일까요?",
                "학원 숙제 앞에서 딴 데를 보는 정도면 치료는 필요 없는 수준일까요?",
                "학원 숙제 앞에서 딴 데를 보는 정도면 정신건강의학과에 가 봐야 하나요?",
                "학원 숙제 앞에서 딴 데를 보는 정도면 소아청소년과 진료를 받아야 하나요?",
                "학원 숙제 앞에서 딴 데를 보는 정도면 약 복용이 도움이 될까요?",
                "학원 숙제 앞에서 딴 데를 보는 정도면 진료 예약을 해야 하나요?",
                "학원 숙제 앞에서 딴 데를 보는 정도면 상담센터에 등록해야 하나요?"):
        assert "G6" in rules(check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 0, bad))), bad
    assert "G6" in rules(check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 0, "딴 데를 자주 보는 날이 이어지면 병원에 데려가야 하는 단계인가요?")))
    for ok in ("궁금한 점은 예약된 상담에서 상담사와 이야기해 보면 되는 것으로 알면 될까요?",
               "상담 예약 후 상담 전까지 학원 숙제 앞에서 딴 데를 보는 모습을 어떻게 적어 두면 될까요?",
               "학원 숙제 앞에서 딴 데를 보는 모습에 대한 약속을 아이와 어떻게 정하면 될까요?"):
        assert "G6" not in rules(check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 0, ok))), ok


# ---------------------------------------------------------------- G3 수치 금지

def test_g3_arabic_digits_are_banned_even_if_they_match_input(prep_out):
    """새 G3: 입력값 그대로인 T점수(주의집중 67)라도 문장에 숫자가 있으면 위반. 수치는 카드가 보여준다."""
    vs = check_output(PROFILE, "prep", _q(prep_out, 0, "주의집중 척도(T점수 67)가 준임상 범위로 보고된 것은 어떻게 읽으면 될까요?"))
    assert "G3" in rules(vs) and any("아라비아 숫자" in v.matched for v in vs)
    assert "G3" in rules(check_output(PROFILE, "prep", _q(copy.deepcopy(FIXTURE["prep"]["attempts"][0]), 0,
                                                          "상담까지 남은 5일 동안 집중이 오래 유지된 활동을 적어 두면 도움이 되나요?")))
    assert "G3" in rules(check_output(PROFILE, "prep", _q(copy.deepcopy(FIXTURE["prep"]["attempts"][0]), 1,
                                                          "결과지에 적힌 상위 15%라는 숫자는 어떤 뜻인가요?")))


@pytest.mark.parametrize("bad", [
    "위축 척도가 일흔다섯 점을 넘는다는 것은 어떤 뜻인가요?",
    "주의집중 척도가 육십칠점이라는 것은 어떤 뜻으로 읽으면 되나요?",
    "위축 척도가 예순 T를 넘었다는 것은 어떤 뜻인가요?",
    "주의집중 척도가 칠십 점 가까이라는 것은 어떤 뜻인가요?",
    # 점수 어휘가 앞에 오는 한국어 자연 어순, 조각 사이 공백, 단독 순우리말 십 단위
    "주의집중 척도의 T점수가 육십칠이라는 것은 어떻게 읽으면 될까요?",
    "주의집중 척도 T점수 예순일곱은 어떻게 읽으면 될까요?",
    "주의집중 척도가 예순일곱이면 준임상 범위에서 어느 위치인가요?",
    "주의집중 척도가 백분위 구십이라는 것은 어떻게 읽으면 될까요?",
    "주의집중 척도가 육십 칠 점이라는 것은 어떻게 읽으면 될까요?",
    "주의집중 척도 결과가 일흔 다섯이라는 것은 어떻게 읽으면 될까요?",
])
def test_g3_korean_numerals_around_score_words_are_banned(prep_out, bad):
    """Codex 점검의 '일흔다섯 점'과 #1의 '육십칠점': 점/T 앞 한글 수사, 점수 어휘 뒤 수사, 단독 십 단위 수사 모두 위반."""
    vs = check_output(PROFILE, "prep", _q(prep_out, 1, bad))
    assert "G3" in rules(vs) and any("한글 수사" in v.matched for v in vs)


@pytest.mark.parametrize("ok", [
    "숙제 앞에서 딴 데를 보는 일이 하루 한 번쯤인 것은 이 점에서 어떻게 보면 될까요?",
    "게임을 끄라고 하면 물건을 던진 적이 두 번 있는데 상담에서는 무엇부터 살펴보게 되나요?",
    "학원 숙제를 앞에 두면 딴 데를 자주 보는 모습을 매일 점심 무렵에도 보는데 어떻게 보면 될까요?",
    "학원 숙제를 앞에 두면 딴 데를 자주 보는 것이 특별한 점인지 상담에서 들을 수 있을까요?",
    # '열', '쉰'은 동사 어간과 겹치므로 단독으로는 수사로 보지 않는다
    "학원 숙제를 앞에 두면 딴 데를 보는 결과가 열어 주는 다음 단계는 무엇인가요?",
    "학원 숙제를 앞에 두면 딴 데를 보는 아이가 잠시 쉰 뒤에 다시 앉으면 달라지나요?",
    "학원 숙제를 앞에 두면 딴 데를 보는 점수 결과가 열린 질문으로 이어지는 이유는 무엇인가요?",
])
def test_g3_ordinary_korean_words_are_not_numerals(prep_out, ok):
    """'이 점', '두 번', '점심', '특별한 점', '열어', '쉰 뒤'처럼 수사가 아닌 흔한 음절은 잡지 않는다 (오검출 방지)."""
    assert "G3" not in rules(check_output(PROFILE, "prep", _q(prep_out, 0, ok)))


# ---------------------------------------------------------------- G4 / G9

def test_g4_invalid_source_scale(prep_out):
    assert "G4" in rules(check_output(PROFILE, "prep", _q(prep_out, 0, prep_out["questions_for_counselor"][0]["question"], "focus_ability")))


def test_g9_normal_scale_cannot_anchor_question_or_observation(prep_out):
    """실LLM 실측 결함: 정상 범위 척도(신체증상 T=51)에 대해 증상이 있는 것처럼 묻는 질문."""
    prep_out["questions_for_counselor"][0] = {
        "question": "신체 증상이 언제 어떤 상황에서 나타나는지 상담에서 어떻게 살펴보게 되나요?",
        "source_scale": "somatic"}
    assert "G9" in rules(check_output(PROFILE, "prep", prep_out))
    prep_out = copy.deepcopy(FIXTURE["prep"]["attempts"][0])
    prep_out["questions_for_counselor"][1]["source_scale"] = "total_problems"  # 종합지표도 포함
    assert "G9" in rules(check_output(PROFILE, "prep", prep_out))


def test_g9_all_normal_profile_allows_only_total_problems():
    """전 척도 정상 프로파일(P1)의 근거는 total_problems뿐이다 (프롬프트 계약과 코드 일치, Codex 지적)."""
    profile = load_profile(ROOT / "data/profiles/p1_all_normal.json")
    out = json.loads((ROOT / "data/fixtures/p1_all_normal.json").read_text(encoding="utf-8"))["prep"]["attempts"][0]
    assert all(it["source_scale"] == "total_problems" for it in out["questions_for_counselor"])
    assert check_output(profile, "prep", out) == []
    out["questions_for_counselor"][0]["source_scale"] = "anxious_depressed"
    assert "G9" in rules(check_output(profile, "prep", out))


# ---------------------------------------------------------------- G5 스키마와 문형

def test_g5_schema_count_and_types(prep_out):
    prep_out["questions_for_counselor"] = prep_out["questions_for_counselor"][:2]
    assert "G5" in rules(check_output(PROFILE, "prep", prep_out))
    prep_out = copy.deepcopy(FIXTURE["prep"]["attempts"][0])
    prep_out["questions_for_counselor"][1] = "그냥 문자열 항목"
    assert "G5" in rules(check_output(PROFILE, "prep", prep_out))
    prep_out = copy.deepcopy(FIXTURE["prep"]["attempts"][0])
    del prep_out["questions_for_counselor"]
    assert "G5" in rules(check_output(PROFILE, "prep", prep_out))


@pytest.mark.parametrize("bad, why", [
    ("학원 숙제를 앞에 두면 딴 데를 자주 보는 모습이 주의집중 척도 결과와 이어지는지 궁금합니다.", "평서문"),
    ("학원 숙제를 앞에 두면 딴 데를 자주 보는 모습은 어떻게 보면 될까요? 그리고 다음 단계는 무엇인가요?", "두 문장"),
    ("숙제 앞에서 딴 데를 보는 것은 왜인가요?", "25자 미만"),
    ("놀이터에서 또래에게 먼저 말을 거는 일이 줄어든 것을 상담에서는 무엇부터 살펴보게 되고, 그 과정에서 보호자가 미리 준비해 두면 좋은 자료나 기록은 어떤 것이 있으며 상담은 보통 몇 회 정도 이어지나요?", "90자 초과"),
    ("학원 숙제를 앞에 두면 딴 데를 자주 보는 모습은 어떻게 보면 될까요? (scale_id: 'attention')", "물음표 뒤 꼬리"),
])
def test_g5_question_form_violations(prep_out, bad, why):
    assert "G5" in rules(check_output(PROFILE, "prep", _q(prep_out, 0, bad))), why


@pytest.mark.parametrize("ok", [
    "학원 숙제를 앞에 두면 딴 데를 자주 보는 모습은 어떻게 보면 될까요?",
    "학원 숙제를 앞에 두면 딴 데를 자주 보는 모습은 상담에서 무엇부터 살펴보게 됩니까?",
    "학원 숙제를 앞에 두면 딴 데를 자주 보는 모습, 이건 집중의 문제로 보는 게 맞죠?",
])
def test_g5_question_form_accepts_interrogative_endings(prep_out, ok):
    assert "G5" not in rules(check_output(PROFILE, "prep", _q(prep_out, 0, ok)))


def test_g5_counts_sentences_outside_quotes_only(prep_out):
    """보호자 의견을 마침표까지 원문 그대로 「」로 인용한 질문은 1문장이다 (TemplateMock과 실 LLM 모두 이렇게 인용한다)."""
    quoted_q = "「학원 숙제를 앞에 두면 딴 데를 자주 봅니다.」라고 적으셨는데, 이 모습은 주의집중 척도의 준임상 결과와 이어서 보면 될까요?"
    assert check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 0, quoted_q)) == []
    quoted_q2 = "\"학원 숙제를 앞에 두면 딴 데를 자주 봅니다.\" 이 모습은 상담에서 무엇부터 살펴보게 되나요?"
    assert "G5" not in rules(check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 0, quoted_q2)))
    # 따옴표 밖의 두 번째 문장은 여전히 위반이다
    two = "「학원 숙제를 앞에 두면 딴 데를 자주 봅니다.」라고 적으셨습니다. 이 모습은 어떻게 보면 될까요?"
    assert "G5" in rules(check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 0, two)))


# ---------------------------------------------------------------- G7 / G8

def test_g7_scale_id_leak_in_question(prep_out):
    """실LLM 실측 결함: 질문 본문에 '(scale_id: 'attention')' 누출."""
    prep_out["questions_for_counselor"][0]["question"] += " (scale_id: 'attention')"
    assert "G7" in rules(check_output(PROFILE, "prep", prep_out))
    assert "G7" in rules(check_output(PROFILE, "prep", _q(copy.deepcopy(FIXTURE["prep"]["attempts"][0]), 1,
                                                          "위축(withdrawn) 척도가 준임상 범위라는 것은 어떤 뜻인가요?")))


def test_g7_uppercase_abbreviations_are_not_leaks(prep_out):
    ok = "학원 숙제를 앞에 두면 딴 데를 자주 보는 모습을 K-CBCL 결과와 SEM 오차 범위로 어떻게 읽으면 될까요?"
    assert "G7" not in rules(check_output(PROFILE, "prep", _q(prep_out, 0, ok)))


def test_g8_nonstandard_band_vocab(prep_out):
    for bad in ("주의집중 척도가 경계 수준으로 나타났다는 것은 어떻게 읽으면 될까요?",
                "주의집중 척도가 경계성(borderline) 영역이라는 것은 어떻게 읽으면 될까요?",
                "주의집중 척도가 위험군에 해당한다는 것은 어떻게 읽으면 될까요?",
                "주의집중 척도가 다소 높은 편이라는 것은 어떻게 읽으면 될까요?",
                "주의집중 척도가 경계군이라는 것은 어떻게 읽으면 될까요?"):
        assert "G8" in rules(check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 0, bad))), bad


def test_g8_label_mismatch_uses_mentioned_or_own_scale(prep_out):
    assert "G8" in rules(check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 0,
                                                          "주의집중 척도가 정상 범위라고 들었는데, 숙제 앞에서 딴 데를 자주 보는 모습은 어떻게 보면 될까요?")))
    # 척도 언급이 없으면 항목의 source_scale(attention, 준임상)과 대조한다
    assert "G8" in rules(check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 0,
                                                          "이 결과가 정상 범위라면 숙제 앞에서 딴 데를 보는 모습은 무엇을 더 살펴보면 될까요?")))
    # 일반어 용법('경계를', '임상적')은 라벨이 아니다
    assert "G8" not in rules(check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 0,
                                                              "이 검사로 알 수 있는 것과 없는 것의 경계를 어디에 두면 되는지 임상적 판단은 어떻게 하나요?")))


# ---------------------------------------------------------------- G10 근거 강제

def test_quotes_caregiver_note_matches_six_contiguous_chars_ignoring_spaces():
    notes = ["학원 숙제를 앞에 두면 딴 데를 자주 봅니다"]
    assert quotes_caregiver_note("숙제를앞에두면 어쩌구", notes)          # 공백 무시, 연속 6자
    assert quotes_caregiver_note("학원 숙제를 앞에 두면 딴 곳을 봅니다", notes)
    assert not quotes_caregiver_note("숙제 앞에서 딴 데를 보는 모습", notes)   # 5자 이하 조각뿐
    assert not quotes_caregiver_note("집중력 저하가 보입니다", notes)


def test_g10_item_passes_with_quote_or_elevated_source_and_fails_with_neither(prep_out):
    # 인용은 있고 근거 척도는 정상 → (a)로 통과 (G9는 따로 걸린다)
    quoted_normal = _q(copy.deepcopy(prep_out), 0, "학원 숙제를 앞에 두면 딴 데를 자주 보는 모습은 어떻게 보면 될까요?", "somatic")
    vs = check_output(PROFILE, "prep", quoted_normal)
    assert "G10" not in rules(vs) and "G9" in rules(vs)
    # 인용은 없고 근거 척도는 준임상 → (b)로 통과
    generic_elevated = _q(copy.deepcopy(prep_out), 0, "이런 검사는 보통 얼마 간격으로 다시 해 보는 것이 좋은가요?", "attention")
    assert "G10" not in rules(check_output(PROFILE, "prep", generic_elevated))
    # 둘 다 아님 → G10 근거 없음
    neither = _q(copy.deepcopy(prep_out), 0, "이런 검사는 보통 얼마 간격으로 다시 해 보는 것이 좋은가요?", "somatic")
    vs = check_output(PROFILE, "prep", neither)
    assert "G10" in rules(vs) and any("근거 없음" in v.matched for v in vs)
    neither_generic = _q(copy.deepcopy(prep_out), 1, "아이가 즐거워한 순간을 적어 두는 것이 상담에 도움이 되나요?", "total_problems")
    assert "G10" in rules(check_output(PROFILE, "prep", neither_generic))


def test_g10_scale_name_in_text_must_match_source_scale(prep_out):
    """Codex 점검의 G4·G9 빈틈: 유효하지만 다른 척도를 source_scale에 붙이는 경우."""
    vs = check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 0,
                                          "학원 숙제를 앞에 두면 딴 데를 자주 보는 모습은 위축 척도 결과와 연결해서 보면 될까요?"))
    assert "G10" in rules(vs) and any("척도 불일치" in v.matched for v in vs)
    vs = check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 2, "주의집중 척도와 관련해 숙제 중 자리를 뜬 횟수를 적어 두면 도움이 될까요?"))
    assert "G10" in rules(vs)
    # 본문은 정상 척도(신체증상)를 말하고 source는 준임상(attention): G9는 못 잡지만 G10이 잡는다
    vs = check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 0,
                                          "신체 증상이 언제 어떤 상황에서 나타나는지 상담에서 어떻게 살펴보게 되나요?", "attention"))
    assert "G10" in rules(vs) and "G9" not in rules(vs)
    # 같은 척도를 언급하면 통과
    assert "G10" not in rules(check_output(PROFILE, "prep", _q(copy.deepcopy(prep_out), 0,
                                                               "학원 숙제를 앞에 두면 딴 데를 자주 보는 모습은 주의집중 척도 결과와 연결해서 보면 될까요?")))


def test_g10_quote_claim_requires_real_fragment(prep_out):
    """'적어 주셨는데'류 인용 주장과 따옴표 인용은 원문 조각이 있어야 한다 (지어낸 관찰 차단)."""
    fabricated = _q(copy.deepcopy(prep_out), 1, "밤마다 운다고 적어 주셨는데, 이 모습을 상담에서는 무엇부터 살펴보게 되나요?")
    vs = check_output(PROFILE, "prep", fabricated)
    assert "G10" in rules(vs) and any("인용 주장" in v.matched for v in vs)
    paraphrased_quote = _q(copy.deepcopy(prep_out), 1, "\"또래와 잘 어울리지 못한다\"라고 적으셨는데 상담에서는 무엇부터 살펴보나요?")
    assert "G10" in rules(check_output(PROFILE, "prep", paraphrased_quote))
    real = _q(copy.deepcopy(prep_out), 1, "놀이터에서 또래에게 먼저 말을 거는 일이 줄었다고 적어 주셨는데, 무엇부터 살펴보게 되나요?")
    assert "G10" not in rules(check_output(PROFILE, "prep", real))


@pytest.mark.parametrize("fabricated", [
    "매일 밤 운다고 하셨는데, 이 모습을 상담에서는 무엇부터 살펴보게 되나요?",
    "매일 밤 운다고 쓰셨는데, 이 모습을 상담에서는 무엇부터 살펴보게 되나요?",
    "매일 밤 우는 모습을 보셨다고 했는데, 상담에서는 무엇부터 살펴보게 되나요?",
    "매일 밤 운다고 말씀하셨는데, 이 모습을 상담에서는 무엇부터 살펴보게 되나요?",
    "밤마다 우는 모습을 관찰하셨듯이 이 결과는 상담에서 어떻게 읽히나요?",
    "‘매일 밤 운다’는 모습은 상담에서 무엇부터 살펴보게 되나요?",
    "『매일 밤 운다』는 모습은 상담에서 무엇부터 살펴보게 되나요?",
    "「매일 밤 운다는 모습은 상담에서 무엇부터 살펴보게 되나요?",          # 짝이 안 맞는 따옴표도 인용으로 본다
])
def test_g10_quote_claim_variants_all_require_real_fragment(prep_out, fabricated):
    """'~다고 하셨는데', '~보셨다고', '관찰하셨듯이', ‘’『』 따옴표: 같은 뜻의 자연스러운 변형도 (a) 필수다.
    근거 척도(attention, 준임상)만으로는 통과하지 못한다."""
    vs = check_output(PROFILE, "prep", _q(prep_out, 0, fabricated, "attention"))
    assert "G10" in rules(vs) and any("인용 주장" in v.matched for v in vs), fabricated


@pytest.mark.parametrize("gloss", [
    "준임상이라고 하는 범위는 또래보다 조금 더 자주 보고된 범위라는 뜻으로 이해하면 될까요?",
    "준임상이라고 부르는 범위는 또래보다 조금 더 자주 보고된 범위라는 뜻으로 이해하면 될까요?",
    "“준임상”이라는 말은 또래보다 조금 더 자주 보고된 범위라는 뜻으로 이해하면 될까요?",
    "\"주의집중 척도\"가 준임상이라는 것은 또래보다 자주 보고되었다는 뜻으로 이해하면 될까요?",
    "결과지에서 'T점수'라고 하는 것은 또래와 견준 위치라는 뜻으로 이해하면 될까요?",
])
def test_g10_gloss_phrases_and_term_quotes_are_not_quote_claims(prep_out, gloss):
    """프롬프트가 권장하는 용어 풀이('~이라고 하는', quality.GLOSS_PATTERNS)와 용어를 감싼 따옴표는 인용 주장이 아니다."""
    assert not has_quote_claim(gloss)
    assert "G10" not in rules(check_output(PROFILE, "prep", _q(prep_out, 0, gloss, "attention"))), gloss


def test_scale_mentions_require_context_for_everyday_words():
    """'위축', '비행'은 일상어('위축된', '비행기')와 겹치므로 척도 어휘나 조사가 붙을 때만 척도 언급이다."""
    assert scale_mentions("비행기 소리에 놀라요") == []
    assert scale_mentions("친구 앞에서 위축된 모습") == []
    assert scale_mentions("위축감이 드는지, 비행시간이 긴지") == []
    assert scale_mentions("위축 척도") == [(0, "withdrawn")]
    assert scale_mentions("비행이 준임상") == [(0, "delinquent")]
    assert scale_mentions("위축과 우울/불안") == [(0, "withdrawn"), (4, "anxious_depressed")]
    assert scale_mentions("위축, 비행 결과") == [(0, "withdrawn"), (4, "delinquent")]
    # 실 LLM 산출물(gemma4:12b p2)에서 본 일상어 '위축된'이 척도 불일치로 막히지 않는다
    out = copy.deepcopy(FIXTURE["prep"]["attempts"][0])
    out["questions_for_counselor"][2] = {"question": "아이가 말수가 적어지거나 위축된 듯한 행동을 보이는 상황은 상담에서 어떻게 보게 되나요?",
                                         "source_scale": "anxious_depressed"}
    assert check_output(PROFILE, "prep", out) == []
    # 보호자 의견에 '비행기'가 있어도 TemplateMock의 attempt 0이 통째로 재생성되지 않는다
    profile = PROFILE.model_copy(update={"caregiver_notes": ["비행기 소리만 나면 깜짝 놀라 웁니다", PROFILE.caregiver_notes[0]]})
    result = run_with_guardrails(profile, "prep", lambda a, p, f: TemplateMockClient().generate("prep", profile, a, "", ""))
    assert result.regen_count == 0 and result.violations == []
    # 척도명으로 쓴 '위축', '비행'은 여전히 source_scale과 대조한다
    vs = check_output(PROFILE, "prep", _q(copy.deepcopy(FIXTURE["prep"]["attempts"][0]), 0,
                                          "학원 숙제를 앞에 두면 딴 데를 자주 보는 모습은 비행 척도와 관련이 있을까요?"))
    assert "G10" in rules(vs) and any("척도 불일치" in v.matched for v in vs)


def test_g10_example_phrase_only_allowed_when_caregiver_wrote_it(prep_out):
    """예시 오염: 프롬프트 예시의 관찰(학원 숙제)은 그 말을 쓴 보호자(p2)에게만 인용할 수 있다."""
    assert "G10" not in rules(check_output(PROFILE, "prep", prep_out))   # p2 의견에 '학원 숙제'가 있다
    p5a = load_profile(ROOT / "data/profiles/p5a_paired_notes.json")
    out = json.loads((ROOT / "data/fixtures/p5a_paired_notes.json").read_text(encoding="utf-8"))["prep"]["attempts"][0]
    assert "G10" not in rules(check_output(p5a, "prep", out))
    out["questions_for_counselor"][0]["question"] = "학원 숙제 앞에서 딴 곳을 자주 보는 모습은 위축 척도의 준임상 결과와 관련이 있을까요?"
    assert "G10" in rules(check_output(p5a, "prep", out))


def test_g10_quote_check_sees_masked_alias():
    """LLM은 이름이 '아이'로 마스킹된 의견을 받으므로, 마스킹본 조각 인용도 정당한 인용이다."""
    profile = PROFILE.model_copy(update={"caregiver_notes": ["김샘플이 학원 숙제를 앞에 두면 딴 데를 자주 봅니다",
                                                            PROFILE.caregiver_notes[1]]})
    out = copy.deepcopy(FIXTURE["prep"]["attempts"][0])
    out["questions_for_counselor"][0]["question"] = "아이가 학원 숙제를 앞에 두면 딴 데를 자주 본다고 적어 주셨는데, 무엇부터 살펴보게 되나요?"
    assert "G10" not in rules(check_output(profile, "prep", out))


# ---------------------------------------------------------------- G11 질문 방향

@pytest.mark.parametrize("bad", [
    "위축이 관찰되는 상황이나 행동 사례를 몇 가지 더 알려주시겠어요?",
    "숙제할 때 딴 데를 보는 모습이 언제 나타나는지 구체적으로 말씀해 주시겠어요?",
    "숙제 앞에서 딴 데를 보는 모습을 관찰하신 구체적인 사례가 있으신가요?",
    "보호자님께서 느끼신 변화 중 가장 먼저 눈에 띈 것은 무엇이었나요?",
])
def test_g11_reverse_direction_questions_are_blocked(prep_out, bad):
    """실LLM 실측 결함 (a): 상담사가 보호자에게 되묻는 문형. 옛 WARN에서 차단으로 승격 (Codex 지적)."""
    vs = check_output(PROFILE, "prep", _q(prep_out, 1, bad, "withdrawn"))
    assert "G11" in rules(vs)


@pytest.mark.parametrize("ok", [
    "놀이터에서 또래에게 먼저 말을 거는 일이 줄어든 것을 상담에서는 무엇부터 살펴보게 되나요?",
    "또래에게 먼저 말을 거는 일이 줄어든 것을 어떻게 읽어야 하는지 설명해 주실 수 있나요?",
    "또래에게 먼저 말을 거는 일이 줄어든 것과 관련해 제가 무엇을 적어 두면 되는지 알려 주실 수 있나요?",
])
def test_g11_caregiver_to_counselor_requests_pass(prep_out, ok):
    """보호자가 상담사에게 하는 요청문은 방향이 같으므로 차단하지 않는다 (양방향 표현은 quality WARN)."""
    assert "G11" not in rules(check_output(PROFILE, "prep", _q(prep_out, 1, ok, "withdrawn")))


# ---------------------------------------------------------------- G12 위기 어휘 출력


@pytest.mark.parametrize("bad", [
    "아이가 없어지고 싶다고 말한 날의 앞뒤 상황을 상담에서 어떻게 다루게 되나요?",   # 외부 리뷰 재현 사례
    "손목을 긋고 싶다는 말이 나오면 상담 전까지 무엇을 하면 되나요?",
    "아이가 죽고 싶다는 말을 하면 상담에서는 어떻게 이어 가게 되나요?",
    "아빠가 때린다고 아이가 말하면 그 자리에서 무엇을 확인하게 되나요?",
])
def test_g12_crisis_vocabulary_in_llm_output_is_blocked(prep_out, bad):
    """입력(p2 의견)에 없는 위기 표현을 모델이 질문에 넣으면 G12. 근거 척도가 유효해 G10으로는 안 잡히는 자리다."""
    vs = check_output(PROFILE, "prep", _q(prep_out, 2, bad, "anxious_depressed"))
    assert "G12" in rules(vs) and any("위기 어휘" in v.matched for v in vs), bad
    assert detect_crisis_signals(PROFILE.model_copy(update={"caregiver_notes": [bad]}))   # 입력 게이트와 같은 사전


def test_g12_shares_dictionary_with_input_gate_and_passes_ordinary_text():
    for text in ("동생을 때려요라고 적으셨는데 상담에서는 무엇부터 살펴보게 되나요?",
                 "때려치우고 싶다는 말을 자주 하는데 어떻게 보면 될까요?",
                 "학원 숙제를 앞에 두면 딴 데를 자주 보는 모습은 어떻게 보면 될까요?"):
        assert _check_crisis_vocab("q", text) == [], text
        assert detect_crisis_signals(PROFILE.model_copy(update={"caregiver_notes": [text]})) == []


def test_g12_persisting_violation_falls_back_to_safe_question(prep_out):
    """위기 어휘가 재생성 2회 뒤에도 남으면 질문 블록은 안전 문구로 대체되고, 최종 출력에 위기 어휘가 없다."""
    bad = _q(prep_out, 2, "아이가 없어지고 싶다고 말한 날의 앞뒤 상황을 상담에서 어떻게 다루게 되나요?")
    seen = []

    def gen_fn(attempt, pending, feedback):
        seen.append((attempt, tuple(pending), {v.rule_id for v in feedback}))
        return copy.deepcopy(bad)

    result = run_with_guardrails(PROFILE, "prep", gen_fn)
    assert [a for a, _p, _f in seen] == [0, 1, 2]
    assert all("G12" in f for _a, _p, f in seen[1:])                      # 재생성 피드백에 G12가 실린다
    assert result.fallback_blocks == ["questions_for_counselor"]
    assert result.output["questions_for_counselor"][0]["question"] == SAFE_QUESTION
    assert _check_crisis_vocab("q", json.dumps(result.output, ensure_ascii=False)) == []
    assert check_output(PROFILE, "prep", result.output) == []
    regen = build_user_message(PROFILE, 1, ["questions_for_counselor"],
                               [Violation("G12", "questions_for_counselor", "없어지고 싶", 0)])
    assert "위기 어휘" in regen                                            # RULE_HINTS에 G12가 있다


# ---------------------------------------------------------------- 루프

def test_fallback_flag_cannot_bypass_checks(prep_out):
    """LLM 출력이 _fallback을 흉내 내도 검사를 우회하지 못한다."""
    prep_out["questions_for_counselor"][0] = {"question": "지금 상태는 심각한 수준인지 상담에서 어떻게 확인하나요?",
                                              "source_scale": "attention", "_fallback": True}

    result = run_with_guardrails(PROFILE, "prep", lambda a, p, f: copy.deepcopy(prep_out))
    assert result.fallback_blocks == ["questions_for_counselor"]
    assert check_output(PROFILE, "prep", result.output) == []


def test_pending_flag_from_llm_output_is_stripped(prep_out):
    """_pending은 생성 전 미리보기 전용 표식이다. LLM 출력에 섞여 오면 화면의 '생성 대기' 태그와 근거 배지 숨김을
    위조하므로 _fallback과 같은 입구에서 벗긴다."""
    prep_out["questions_for_counselor"][0]["_pending"] = True
    prep_out["questions_for_counselor"][1]["_pending"] = "yes"
    stripped = _strip_fallback_flags(prep_out)
    assert not any("_pending" in it or "_fallback" in it for it in stripped["questions_for_counselor"])
    result = run_with_guardrails(PROFILE, "prep", lambda a, p, f: copy.deepcopy(prep_out))
    assert result.violations == [] and result.fallback_blocks == []
    assert not any("_pending" in it for block in result.output.values() for it in block)


def test_fail_closed_fallback_after_two_regens_with_feedback(prep_out):
    """계속 위반하는 블록은 재생성 2회 후 안전 문구로 대체되고, 최종 출력에는 잔존 위반이 없어야 한다."""
    prep_out["questions_for_counselor"][0]["question"] += " (scale_id: 'attention')"
    calls = []

    def gen_fn(attempt, pending, feedback):
        calls.append(attempt)
        if attempt > 0:
            assert pending == ["questions_for_counselor"]
            assert any(v.rule_id == "G7" for v in feedback)
        return copy.deepcopy(prep_out)

    result = run_with_guardrails(PROFILE, "prep", gen_fn)
    assert calls == [0, 1, 2] and result.regen_count == 2
    assert result.fallback_blocks == ["questions_for_counselor"]
    assert list(result.output) == ["questions_for_counselor"]
    assert check_output(PROFILE, "prep", result.output) == []
