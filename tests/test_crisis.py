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


class NeverCallClient:
    """호출되면 즉시 실패하는 클라이언트 - LLM 미호출의 증명용."""

    def generate(self, *args, **kwargs):
        raise AssertionError("위기 프로파일에서 LLM이 호출되었습니다")


def test_crisis_keywords_detected_in_c1():
    hits = detect_crisis_signals(CRISIS)
    assert hits, "c1_crisis의 긴급 키워드가 검출되어야 한다"


@pytest.mark.parametrize("path", OTHERS, ids=[p.stem for p in OTHERS])
def test_no_false_positive_on_other_profiles(path):
    assert detect_crisis_signals(load_profile(path)) == []


def test_generate_refuses_crisis_profile_without_llm_call():
    with pytest.raises(CrisisSignalDetected):
        generate_all(CRISIS, NeverCallClient())


def test_crisis_html_contains_help_lines_and_no_scores():
    html = build_crisis_html(CRISIS)
    for needed in ("109", "1577-0199", "1388", "자동 해설"):
        assert needed in html
    # 위기 화면에는 점수 해설이 없어야 하고, 검출된 원문도 재노출하지 않는다
    assert "T점수" not in html
    assert "죽고 싶다" not in html
