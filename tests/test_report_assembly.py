"""결정론 조립 블록 (ADR 0010): 연결 문단(build_overview_text)과 상담사에게 전달할 요약
미리보기(build_counselor_briefing).

두 함수는 순수 함수이며 LLM 출력이 아니다. 이 테스트는 (1) 설계 노트의 문구(보호자 의견 원문
큰따옴표 인용, 상승 척도는 보고서 라벨 그대로, 전 척도 정상 분기), (2) 렌더된 HTML의 "결정론 조립"
태그 2개와 새 라벨, (3) 이 텍스트가 LLM 게이트(가드레일·품질 지표)의 입력에 섞이지 않음을 고정한다.
"""

import html as html_lib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import generate_all
from src.guardrails import TASK_BLOCKS, check_output, run_with_guardrails
from src.llm_client import make_client
from src.parser import load_profile
from src.quality import caregiver_texts, quality_summary
from src.report_html import (ASSEMBLED_TAG, BRIEFING_LABEL, LLM_TAG, OVERVIEW_LABEL,
                             build_counselor_briefing, build_overview_text,
                             build_pending_report_html, build_report_html)


def _profile(name):
    return load_profile(ROOT / "data/profiles" / f"{name}.json")


def _unescape(html: str) -> str:
    return html_lib.unescape(html)


# ---------------------------------------------------------------- 연결 문단

def test_overview_quotes_notes_verbatim_and_lists_elevated_scales_by_report_label():
    p2 = _profile("p2_partial_borderline")
    text = build_overview_text(p2)
    assert text == (
        '보호자님은 이렇게 적어 주셨습니다. "학원 숙제를 앞에 두면 딴 데를 자주 봅니다" '
        '"놀이터에서 또래에게 먼저 말을 거는 일이 줄었습니다" '
        "검사에서는 내재화 문제, 위축, 우울/불안, 주의집중이 준임상 범위로 보고되었고, 그 밖의 척도는 정상 범위였습니다. "
        "이 관찰과 결과가 어떻게 이어지는지는 예약된 상담에서 상담사와 이야기해 보세요."
    )
    assert not re.search(r"\d", text)                       # 수치는 카드가 보여준다


def test_overview_lists_clinical_then_borderline_and_uses_josa():
    text = build_overview_text(_profile("p3_boundary_mix"))
    assert "검사에서는 공격성이 임상 범위로, 총 문제행동, 외현화 문제, 위축, 주의집중이 준임상 범위로 보고되었고, 그 밖의 척도는 정상 범위였습니다." in text
    text = build_overview_text(_profile("p4_clinical"))
    assert "총 문제행동, 외현화 문제, 공격성이 임상 범위로, 주의집중, 비행이 준임상 범위로" in text


def test_overview_all_normal_branch():
    text = build_overview_text(_profile("p1_all_normal"))
    assert text == (
        '보호자님은 이렇게 적어 주셨습니다. "동생이 태어난 뒤로 잠드는 데 시간이 오래 걸립니다" '
        '"정상 범위라는 말을 들어도 마음이 놓이지 않아 여쭤보고 싶습니다" '
        "검사에서는 모든 척도가 정상 범위로 보고되었습니다. 관찰하신 모습이 무엇을 뜻하는지는 상담에서 함께 살펴볼 수 있습니다."
    )
    assert "준임상" not in text and "임상 범위" not in text


def test_overview_without_notes_and_when_unscheduled():
    p2 = _profile("p2_partial_borderline")
    empty = p2.model_copy(update={"caregiver_notes": []})
    text = build_overview_text(empty)
    assert text.startswith("보호자 의견은 따로 적히지 않았습니다. 검사에서는 내재화 문제")
    assert '"' not in text
    p1_empty = _profile("p1_all_normal").model_copy(update={"caregiver_notes": []})
    assert "이 결과가 무엇을 뜻하는지는" in build_overview_text(p1_empty)
    unscheduled = p2.model_copy(update={"counseling_scheduled": False})
    text = build_overview_text(unscheduled)
    assert "예약된 상담" not in text and "상담 예약 후 상담사와 이야기해 보세요" in text


# ---------------------------------------------------------------- 상담사 요약

def test_briefing_has_notes_elevated_table_questions_and_days():
    p2 = _profile("p2_partial_borderline")
    questions = [{"question": "첫 질문인가요?", "source_scale": "attention"}, "둘째 질문인가요?"]
    text = build_counselor_briefing(p2, questions, 5, True)
    lines = text.split("\n")
    assert lines[0] == "[보호자 의견 원문 2건]"
    assert lines[1] == '1. "학원 숙제를 앞에 두면 딴 데를 자주 봅니다"'
    assert "[상승 척도 4개] 척도, T점수, 보고서 라벨" in lines
    assert "- 내재화 문제 T=62 준임상" in lines and "- 주의집중 T=67 준임상" in lines
    assert "[상담사에게 물어볼 질문 2개] 위 목록의 체크 표시 기준" in lines
    assert "1. 첫 질문인가요?" in lines and "2. 둘째 질문인가요?" in lines
    assert lines[-1] == "[상담까지 5일]"


def test_briefing_all_normal_unscheduled_and_no_questions():
    p1 = _profile("p1_all_normal")
    text = build_counselor_briefing(p1, [], 17, False)
    assert "[상승 척도 없음] 모든 척도 정상 범위" in text
    assert "[상담사에게 물어볼 질문] 아직 생성되지 않음" in text
    assert text.endswith("[상담 미예약]") and "상담까지" not in text
    empty = p1.model_copy(update={"caregiver_notes": []})
    assert build_counselor_briefing(empty, [], 3, True).startswith("[보호자 의견 원문] 적히지 않음")


def test_briefing_for_p4_lists_clinical_and_borderline_with_t_scores():
    text = build_counselor_briefing(_profile("p4_clinical"), [], 4, True)
    assert "- 총 문제행동 T=63 임상" in text and "- 공격성 T=71 임상" in text and "- 비행 T=62 준임상" in text


# ---------------------------------------------------------------- 렌더와 게이트 분리

def test_rendered_report_tags_two_assembled_blocks_and_new_briefing_label():
    profile = _profile("p2_partial_borderline")
    results = generate_all(profile, make_client("mock"))
    html = build_report_html(profile, results)
    assert html.count(f'<span class="tag">{ASSEMBLED_TAG}</span>') == 2
    assert f"<h3>{OVERVIEW_LABEL} " in html
    assert BRIEFING_LABEL in html and "보호자 화면에는 표시되지 않는" not in html and "상담사용 사전 요약" not in html
    assert _unescape(build_overview_text(profile)) in _unescape(html)
    briefing = build_counselor_briefing(profile, results["prep"].output["questions_for_counselor"], 5, True)
    assert _unescape(briefing) in _unescape(html)
    assert html.count(LLM_TAG) == 2                                # 질문·관찰 두 블록만 LLM 생성
    assert "생성 문구 · 검증 통과" not in html
    assert "질문과 관찰 포인트는 LLM이, 연결 문단과 상담사 요약은 결정론 조립이" in html
    # 생성 전 미리보기에서도 조립 블록은 실제 문구이고, 요약의 질문 자리만 '아직 생성되지 않음'이다
    pending = build_pending_report_html(profile)
    assert pending.count(f'<span class="tag">{ASSEMBLED_TAG}</span>') == 2
    assert _unescape(build_overview_text(profile)) in _unescape(pending)
    assert "아직 생성되지 않음" in pending and "생성 대기" in pending


def test_assembled_texts_are_outside_llm_gates():
    """조립 텍스트는 LLM 출력에도, 가드레일이 검사하는 블록에도, 품질 지표 입력에도 없다.

    상담사 요약에는 T점수(아라비아 숫자)가 들어 있어 G3 대상이었다면 걸렸을 텍스트다. 그것이 걸리지 않고
    리포트에 그대로 나가는 것이 "게이트 대상이 아님"의 증거다.
    """
    profile = _profile("p2_partial_borderline")
    results = generate_all(profile, make_client("mock"))
    prep = results["prep"]
    overview = build_overview_text(profile)
    briefing = build_counselor_briefing(profile, prep.output["questions_for_counselor"], 5, True)
    assert re.search(r"T=\d+", briefing)                                        # 숫자가 있다
    assert "overview" not in prep.output and "counselor_briefing" not in prep.output
    assert all(b in ("questions_for_counselor", "observation_points") for b in TASK_BLOCKS["prep"])
    gated = " ".join(t for _b, t in caregiver_texts("prep", prep.output))
    assert overview not in gated and briefing not in gated
    assert not re.search(r"\d", gated)                                          # LLM 텍스트에는 숫자가 없다
    assert check_output(profile, "prep", prep.output) == []
    q = quality_summary(profile, {"prep": prep.output})
    assert q["jargon"]["blocks_total"] == len(caregiver_texts("prep", prep.output))
    # 가드레일 루프가 어떤 출력을 받아도 조립 텍스트를 만들거나 검사하지 않는다
    seen = []

    def gen_fn(attempt, pending, feedback):
        seen.append(tuple(pending))
        return prep.output

    run_with_guardrails(profile, "prep", gen_fn)
    assert seen == [("questions_for_counselor", "observation_points")]
