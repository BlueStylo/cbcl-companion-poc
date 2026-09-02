"""위기 신호 중단 규칙 테스트 (LLM 호출 없음).

긴급 키워드가 입력에 있으면 LLM 호출 자체가 일어나지 않아야 하고,
화면은 상담 연결 안내만 출력해야 한다 (fail-closed).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import generate_all
from src.guardrails import CrisisSignalDetected, detect_crisis_signals
from src.parser import load_profile
from src.report_html import build_crisis_html

CRISIS = load_profile(ROOT / "data/profiles/c1_crisis.json")
OTHERS = [p for p in sorted((ROOT / "data/profiles").glob("*.json"))
          if p.stem != "c1_crisis"]

CRISIS_POSITIVES = (
    "차라리 죽겠다",
    "차라리 죽겠다고 말한다",
    "사는 이유가 없다",
    "살아갈 이유가 없다고 한다",
    "손목을 긋겠다",
    "손목을 긋고 싶다고 한다",
    "죽고 싶다고 한다",
    "죽고싶다는 말을 했다",
    "없어지고 싶다",
    "없어지고 싶다고 말한다",
    "사라지고 싶어 한다",
    "옥상에서 뛰어내리겠다",
    "창문에서 뛰어내리고 싶다",
    "다리에서 뛰어내리려 했다",
    "목을 매",
    "목을 매겠다고 한다",
    "목을 매려고 했다",
    "약을 한꺼번에 먹겠다",
    "수면제를 한꺼번에 삼켰다",
    "알약을 몽땅 먹고 싶다고 한다",
    "자해 흔적이 보인다",
    "자해 자국을 발견했다",
    "스스로 몸을 해치겠다고 한다",
    "팔에 상처를 냈다",
    "맞아서 멍이 들었다",
    "맞고 상처가 났다고 한다",
    "아빠가 때린다고 한다",
    "선생님이 때리겠다고 했다고 한다",
    "집에서 폭행을 당했다고 한다",
    "학대를 받는다고 말했다",
    "성적인 접촉이 있었다고 한다",
    "원치 않는 성적 접촉을 말했다",
    "성추행을 당했다고 한다",
    "성폭력을 겪었다고 한다",
    "누군가 몸을 만졌다고 한다",
    "옷을 벗기려고 했다고 한다",
)

CRISIS_NEGATIVES = (
    "죽도록 힘들었다",
    "죽을 만큼 피곤하다",
    "게임에서 캐릭터가 죽었다",
    "화분 식물이 죽어서 속상하다",
    "계단 두 칸을 뛰어내렸다",
    "줄넘기하다 손목을 긁었다",
    "연필로 선을 그었다",
    "목을 매일 스트레칭한다",
    "약 봉지를 한꺼번에 정리했다",
    "알약을 한 통에 모아 두었다",
    "친구와 장난치다 어깨를 툭 맞았다",
    "햇빛에 멍하니 서 있었다",
    "성적인 발달을 배우는 수업이었다",
    "접촉 사고가 났다",
    "폭행 장면이 나오는 영화를 봤다",
    "학대 예방 교육을 들었다",
    "옷을 벗고 수영복으로 갈아입었다",
)


class NeverCallClient:
    """호출되면 즉시 실패하는 클라이언트 - LLM 미호출의 증명용."""

    def generate(self, *args, **kwargs):
        raise AssertionError("위기 프로파일에서 LLM이 호출되었습니다")


def test_crisis_keywords_detected_in_c1():
    hits = detect_crisis_signals(CRISIS)
    assert hits, "c1_crisis의 긴급 키워드가 검출되어야 한다"


@pytest.mark.parametrize("text", CRISIS_POSITIVES)
def test_korean_crisis_and_abuse_expressions_are_detected(text):
    profile = CRISIS.model_copy(update={"caregiver_notes": [text]})
    assert detect_crisis_signals(profile), text


@pytest.mark.parametrize("text", CRISIS_NEGATIVES)
def test_crisis_idioms_and_unrelated_contexts_pass(text):
    profile = CRISIS.model_copy(update={"caregiver_notes": [text]})
    assert detect_crisis_signals(profile) == [], text


@pytest.mark.parametrize("path", OTHERS, ids=[p.stem for p in OTHERS])
def test_no_false_positive_on_other_profiles(path):
    assert detect_crisis_signals(load_profile(path)) == []


@pytest.mark.parametrize("text", [
    "차라리 죽겠다",
    "손목을 긋겠다",
    "맞아서 멍이 들었다",
    "성적인 접촉이 있었다고 한다",
])
def test_generate_refuses_expanded_crisis_profile_without_llm_call(text):
    profile = CRISIS.model_copy(update={"caregiver_notes": [text]})
    with pytest.raises(CrisisSignalDetected):
        generate_all(profile, NeverCallClient())


def test_crisis_html_contains_help_lines_and_no_scores():
    html = build_crisis_html(CRISIS)
    for needed in ("109", "1577-0199", "1388", "자동 해설"):
        assert needed in html
    # 위기 화면에는 점수 해설이 없어야 하고, 검출된 원문도 재노출하지 않는다
    assert "T점수" not in html
    assert "죽고 싶다" not in html
