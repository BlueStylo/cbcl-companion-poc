"""결정론 조립 블록 (ADR 0010과 그 보강): 연결 문단(build_overview_text), 가정 관찰 포인트
(build_observation_points), 상담사에게 전달할 요약 미리보기(build_counselor_briefing).

세 함수는 순수 함수이며 LLM 출력이 아니다. 이 테스트는 (1) 설계 노트의 문구(보호자 의견 원문
큰따옴표 인용, 상승 척도는 보고서 라벨 그대로, 전 척도 정상 분기, 관찰 포인트의 위계 순서 선택),
(2) 렌더된 HTML의 "결정론 조립" 태그 3개와 새 라벨, (3) 이 텍스트가 LLM 게이트(가드레일·품질 지표)의
입력에 섞이지 않음을 고정한다.
"""

import html as html_lib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import generate_all
from src.guardrails import TASK_BLOCKS, _check_crisis_vocab, _check_text, check_output, run_with_guardrails
from src.llm_client import make_client
from src.parser import COMPOSITE_IDS, SYNDROME_IDS, load_profile
from src.quality import caregiver_texts, quality_summary
from src.report_html import (ASSEMBLED_TAG, BRIEFING_LABEL, BRIEFING_QUESTIONS_NOTE, FALLBACK_TAG, LLM_TAG,
                             OVERVIEW_LABEL, UNSCHEDULED_LABEL, build_counselor_briefing,
                             build_observation_points, build_overview_text, build_pending_report_html,
                             build_report_html)
from src.scale_texts import GENERAL_OBSERVATION_TEXTS, OBSERVATION_TEXT


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
        "검사에서는 내재화 문제, 위축, 우울/불안, 주의집중이 준임상 범위로 보고되었고, 그 밖의 포함 척도는 정상 범위였습니다. "
        "이 관찰과 결과가 어떻게 이어지는지는 예약된 상담에서 상담사와 이야기해 보세요."
    )
    assert not re.search(r"\d", text)                       # 수치는 카드가 보여준다


def test_overview_lists_clinical_then_borderline_and_uses_josa():
    text = build_overview_text(_profile("p3_boundary_mix"))
    assert "검사에서는 공격성이 임상 범위로, 총 문제행동, 외현화 문제, 위축, 주의집중이 준임상 범위로 보고되었고, 그 밖의 포함 척도는 정상 범위였습니다." in text
    text = build_overview_text(_profile("p4_clinical"))
    assert "총 문제행동, 외현화 문제, 공격성이 임상 범위로, 주의집중, 비행이 준임상 범위로" in text


def test_overview_all_normal_branch():
    text = build_overview_text(_profile("p1_all_normal"))
    assert text == (
        '보호자님은 이렇게 적어 주셨습니다. "동생이 태어난 뒤로 잠드는 데 시간이 오래 걸립니다" '
        '"정상 범위라는 말을 들어도 마음이 놓이지 않아 여쭤보고 싶습니다" '
        "검사에서는 이번 가이드에 포함된 척도는 모두 정상 범위로 보고되었습니다. 관찰하신 모습이 무엇을 뜻하는지는 상담에서 함께 살펴볼 수 있습니다."
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


# ---------------------------------------------------------------- 관찰 포인트

def test_observation_points_pick_elevated_scales_in_hierarchy_order():
    """준임상 이상 척도를 위계 순서(종합지표 다음 개별 척도)로 셋 골라 척도별 고정 문구를 붙인다.

    같은 층 안에서는 임상이 준임상보다 먼저다: p4는 개별 척도 중 주의집중(준임상)이 위계상 앞이지만
    공격성(임상)이 뽑힌다. 전부 준임상인 p2는 순수 위계 순서 그대로다.
    """
    p2 = build_observation_points(_profile("p2_partial_borderline"))
    assert [o["source_scale"] for o in p2] == ["internalizing", "withdrawn", "anxious_depressed"]
    assert [o["point"] for o in p2] == [OBSERVATION_TEXT[s] for s in ("internalizing", "withdrawn", "anxious_depressed")]
    p4 = build_observation_points(_profile("p4_clinical"))
    assert [o["source_scale"] for o in p4] == ["total_problems", "externalizing", "aggressive"]
    p3 = build_observation_points(_profile("p3_boundary_mix"))            # 총점·외현화 준임상, 공격성 임상
    assert [o["source_scale"] for o in p3] == ["total_problems", "externalizing", "aggressive"]
    assert all(set(o) == {"point", "source_scale"} for o in p2 + p4)


def test_observation_points_fill_with_general_texts_when_fewer_than_three_elevated():
    """상승 척도가 셋 미만이면 총 문제행동 기준 일반 문구로 채우고, 전 척도 정상이면 일반 문구 셋이 전부다."""
    two = build_observation_points(_profile("c1_crisis"))                  # 내재화, 우울/불안 두 개만 상승
    assert [o["source_scale"] for o in two] == ["internalizing", "anxious_depressed", "total_problems"]
    assert two[2]["point"] == GENERAL_OBSERVATION_TEXTS[0]
    none = build_observation_points(_profile("p1_all_normal"))
    assert [o["point"] for o in none] == list(GENERAL_OBSERVATION_TEXTS)
    assert all(o["source_scale"] == "total_problems" for o in none)


def test_observation_texts_are_nominal_fixed_phrases_without_numbers_or_gated_vocabulary():
    """고정 문구 자체가 계약을 지킨다: 명사형 종결, 숫자 없음, LLM에 금지하는 어휘(G1/G2/G6/G8/G12) 없음, 예시 문구 없음."""
    profile = _profile("p2_partial_borderline")
    assert set(OBSERVATION_TEXT) == set((*COMPOSITE_IDS, *SYNDROME_IDS)) and len(GENERAL_OBSERVATION_TEXTS) == 3
    for text in (*OBSERVATION_TEXT.values(), *GENERAL_OBSERVATION_TEXTS):
        assert text.endswith("기") and not re.search(r"\d", text), text
        vs = [v for v in _check_text("obs", text, profile) if v.rule_id != "G3"]
        assert vs == [] and _check_crisis_vocab("obs", text) == [], (text, [v.matched for v in vs])
        for banned in ("학원 숙제", "놀이터", "또래에게 먼저 말", "높은 편", "심각", "치료"):
            assert banned not in text, (text, banned)


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
    assert f"[상담사에게 물어볼 질문 2개] {BRIEFING_QUESTIONS_NOTE}" in lines
    assert "위 목록의 체크 표시 기준" not in text          # 체크 상태를 읽는 코드가 없던 시절의 문구
    assert "1. 첫 질문인가요?" in lines and "2. 둘째 질문인가요?" in lines
    assert lines[-1] == "[상담까지 5일]"


def test_briefing_question_section_is_wired_to_checkboxes_in_template():
    """요약의 질문 절 머리글("위 목록에서 체크한 질문")은 사실이어야 한다: 템플릿에 질문 목록의 체크박스
    change 이벤트를 받아 요약 텍스트의 같은 번호 줄을 숨기는 스크립트가 있고, 두 요소에 그 스크립트가
    찾는 id가 있다. 생성 시점 텍스트는 전체 질문(체크박스 기본값 전부 체크)이므로 스크립트 없이도 거짓이 아니다."""
    profile = _profile("p2_partial_borderline")
    html = build_report_html(profile, generate_all(profile, make_client("mock")))
    assert '<ul class="qlist" id="question-list">' in html
    assert 'id="brief-text"' in html
    script = html[html.index("<script data-brief-sync>"):]
    script = script[:script.index("</script>")]
    assert "getElementById('question-list')" in script and "getElementById('brief-text')" in script
    assert "addEventListener('change'" in script
    assert "\\[상담사에게 물어볼 질문 \\d+개\\]" in script       # 머리글 형식이 바뀌면 스크립트도 같이 바뀌어야 한다
    # 체크박스 수 = 요약 질문 절의 번호 줄 수 (스크립트가 순서대로 1:1 대응시킨다)
    questions = generate_all(profile, make_client("mock"))["prep"].output["questions_for_counselor"]
    briefing = build_counselor_briefing(profile, questions, 5, True)
    section = briefing[briefing.index("[상담사에게 물어볼 질문"):]
    numbered = [l for l in section.split("\n")[1:] if re.match(r"^\d+\. ", l)]
    assert html.count('<input type="checkbox" checked>') == len(numbered) == len(questions)


def test_briefing_all_normal_unscheduled_and_no_questions():
    """전 척도 정상 요약은 연결 문단과 같은 범위 규칙을 따른다: 특수 척도 미실시면 "이번 가이드에 포함된 척도"."""
    p1 = _profile("p1_all_normal")                                          # special_scales_administered=False
    text = build_counselor_briefing(p1, [], 17, False)
    assert "[상승 척도 없음] 이번 가이드에 포함된 척도는 모두 정상 범위" in text and "모든 척도 정상 범위" not in text
    assert "[상담사에게 물어볼 질문] 아직 생성되지 않음" in text
    assert text.endswith(f"[{UNSCHEDULED_LABEL}]") and "상담까지" not in text and "미예약" not in text
    administered = p1.model_copy(update={"special_scales_administered": True})
    assert "[상승 척도 없음] 모든 척도 정상 범위" in build_counselor_briefing(administered, [], 17, True)
    empty = p1.model_copy(update={"caregiver_notes": []})
    assert build_counselor_briefing(empty, [], 3, True).startswith("[보호자 의견 원문] 적히지 않음")


def test_unscheduled_label_is_identical_on_screen_and_in_briefing():
    """미예약이면 화면 머리글 링크와 요약이 같은 문구("상담 예약 후 사용")를 쓴다."""
    profile = _profile("p2_partial_borderline").model_copy(update={"counseling_scheduled": False, "days_until_counseling": 0})
    html = build_report_html(profile, generate_all(profile, make_client("mock")))
    assert UNSCHEDULED_LABEL == "상담 예약 후 사용"
    assert f"· {UNSCHEDULED_LABEL}</small>" in html and f"[{UNSCHEDULED_LABEL}]" in _unescape(html)
    assert "[상담 미예약]" not in html and "예약된 상담" not in html


def test_fallback_question_block_is_tagged_as_replaced_not_verified():
    """질문 블록이 안전 문구로 대체되면 머리글 태그는 "안전 문구로 대체됨"이고 "LLM 생성 · 검증 통과"는 붙지 않는다."""
    a1 = _profile("a1_adversarial")
    results = generate_all(a1, make_client("mock"))
    assert results["prep"].fallback_blocks == ["questions_for_counselor"]
    html = build_report_html(a1, results)
    assert f'<span class="tag fb">{FALLBACK_TAG}</span>' in html and LLM_TAG not in html
    assert html.count('<span class="tag fb">안전 문구</span>') == 1            # 항목 태그는 그대로
    p2 = _profile("p2_partial_borderline")
    ok = build_report_html(p2, generate_all(p2, make_client("mock")))
    assert FALLBACK_TAG not in ok and ok.count(LLM_TAG) == 1


def test_briefing_for_p4_lists_clinical_and_borderline_with_t_scores():
    text = build_counselor_briefing(_profile("p4_clinical"), [], 4, True)
    assert "- 총 문제행동 T=63 임상" in text and "- 공격성 T=71 임상" in text and "- 비행 T=62 준임상" in text


# ---------------------------------------------------------------- 렌더와 게이트 분리

def test_rendered_report_tags_two_assembled_blocks_and_new_briefing_label():
    profile = _profile("p2_partial_borderline")
    results = generate_all(profile, make_client("mock"))
    html = build_report_html(profile, results)
    assert html.count(f'<span class="tag">{ASSEMBLED_TAG}</span>') == 3          # 연결 문단, 관찰 포인트, 요약
    assert f"<h3>{OVERVIEW_LABEL} " in html
    observations = html[html.index('id="observation-list"'):html.index("</ul>", html.index('id="observation-list"'))]
    assert all(_unescape(o["point"]) in _unescape(observations) for o in build_observation_points(profile))
    assert observations.count("참고 척도:") == 3 and "생성 대기" not in observations and LLM_TAG not in observations
    assert BRIEFING_LABEL in html and "보호자 화면에는 표시되지 않는" not in html and "상담사용 사전 요약" not in html
    assert _unescape(build_overview_text(profile)) in _unescape(html)
    briefing = build_counselor_briefing(profile, results["prep"].output["questions_for_counselor"], 5, True)
    assert _unescape(briefing) in _unescape(html)
    assert html.count(LLM_TAG) == 1                                # 질문 블록만 LLM 생성
    assert "생성 문구 · 검증 통과" not in html
    assert "질문은 LLM이, 연결 문단과 관찰 포인트와 상담사 요약은 결정론 조립이" in html
    # 생성 전 미리보기에서도 조립 블록은 실제 문구이고, 요약의 질문 자리만 '아직 생성되지 않음'이다
    pending = build_pending_report_html(profile)
    assert pending.count(f'<span class="tag">{ASSEMBLED_TAG}</span>') == 3
    assert _unescape(build_overview_text(profile)) in _unescape(pending)
    assert all(_unescape(o["point"]) in _unescape(pending) for o in build_observation_points(profile))
    assert "아직 생성되지 않음" in pending and pending.count("생성 대기") >= 1


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
    assert "overview" not in prep.output and "counselor_briefing" not in prep.output and "observation_points" not in prep.output
    assert TASK_BLOCKS["prep"] == ("questions_for_counselor",)
    gated = " ".join(t for _b, t in caregiver_texts("prep", prep.output))
    assert overview not in gated and briefing not in gated
    assert all(o["point"] not in gated for o in build_observation_points(profile))
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
    assert seen == [("questions_for_counselor",)]


def test_briefing_all_fallback_is_not_counted_and_is_schedule_aware():
    """안전 문구만 남은 질문 절은 개수로 세지 않고, 미예약이면 '예약된 상담' 전제를 쓰지 않는다."""
    from src.guardrails import SAFE_QUESTION
    from src.parser import load_profile
    from src.report_html import FALLBACK_TAG, build_counselor_briefing

    profile = load_profile("data/profiles/p2_partial_borderline.json")
    fallback = [{"question": SAFE_QUESTION, "source_scale": "total_problems", "_fallback": True}]
    scheduled = build_counselor_briefing(profile, fallback, days=5, counseling_scheduled=True)
    assert FALLBACK_TAG in scheduled and "질문 1개" not in scheduled
    unscheduled = build_counselor_briefing(profile, fallback, days=0, counseling_scheduled=False)
    assert "예약된 상담" not in unscheduled
    normal = build_counselor_briefing(profile, [{"question": "이 부분은 상담에서 무엇부터 살펴보게 되나요?",
                                                "source_scale": "attention"}], days=5,
                                      counseling_scheduled=True)
    assert "질문 1개" in normal
