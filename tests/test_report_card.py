"""척도 카드 정보 순서 (ADR 0008): 결론 → 쉬운 구간 + 라벨 배지 + 수치 → 곡선 → 해설 → 진단 아님 한 줄.

카드 상단의 결론·구간 이름·진단 아님 문구는 밴드별 고정 문구라 LLM 출력이 아니고,
따라서 가드레일(G1~G10)과 품질 지표의 대상이 아니어야 한다. 이 테스트는 (1) 렌더된
카드의 순서, (2) 준임상·임상 카드에만 붙는 한 줄, (3) 고정 문구가 게이트 입력에
섞이지 않음을 고정한다.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import generate_all
from src.guardrails import check_output
from src.llm_client import make_client
from src.parser import load_profile
from src.quality import caregiver_texts
from src.report_html import (CARD_NOT_DIAGNOSIS, CARD_PLAIN_RANGE, CARD_VERDICT,
                             CURVE_HOWTO_CAPTION, CURVE_HOWTO_LINES, build_pending_report_html,
                             build_report_html)


def _profile(name):
    return load_profile(ROOT / "data/profiles" / f"{name}.json")


def _card(html: str, scale_id: str) -> str:
    """id="scale-<sid>" 요소부터 다음 카드(또는 절 끝)까지의 HTML 조각."""
    start = html.index(f'id="scale-{scale_id}"')
    nxt = re.compile(r'id="scale-|<h3|<div class="psyedu"|<details class="fold"').search(html, start + 1)
    return html[start:nxt.start()]


def _order(fragment: str, *needles: str) -> list[int]:
    positions = [fragment.find(n) for n in needles]
    assert all(p >= 0 for p in positions), [n for n, p in zip(needles, positions) if p < 0]
    return positions


def test_borderline_card_reads_in_fixed_order():
    """p2 내재화(종합, 62T 준임상): 척도명 → 결론 → 쉬운 구간 → 라벨 배지 → 수치 → 곡선 → 해설 → 진단 아님."""
    html = build_pending_report_html(_profile("p2_partial_borderline"))
    card = _card(html, "internalizing")
    pos = _order(card, "내재화 문제", CARD_VERDICT["borderline"], CARD_PLAIN_RANGE["borderline"],
                 '<span class="chip borderline">준임상</span>', "이번 검사 결과 62T", "<svg",
                 "준임상 범위입니다", CARD_NOT_DIAGNOSIS)
    assert pos == sorted(pos), pos
    assert "이번 결과 62T" in card and "우리 아이" not in card
    assert "오차 범위 58~66T (예시값)" in card


def test_clinical_card_has_not_diagnosis_line_and_normal_cards_do_not():
    p4 = build_pending_report_html(_profile("p4_clinical"))
    aggressive = _card(p4, "aggressive")
    pos = _order(aggressive, "공격성", CARD_VERDICT["clinical"], CARD_PLAIN_RANGE["clinical"],
                 '<span class="chip clinical">임상</span>', "이번 검사 결과 71T", "<svg", CARD_NOT_DIAGNOSIS)
    assert pos == sorted(pos), pos
    # 준임상 2 + 임상 3 = 5개 카드에만 한 줄이 붙는다
    assert p4.count(CARD_NOT_DIAGNOSIS) == 5
    withdrawn = _card(p4, "withdrawn")
    assert CARD_VERDICT["normal"] in withdrawn and CARD_NOT_DIAGNOSIS not in withdrawn

    p1 = build_pending_report_html(_profile("p1_all_normal"))
    assert CARD_NOT_DIAGNOSIS not in p1
    assert p1.count(f'<span class="verdict">{CARD_VERDICT["normal"]}</span>') == 11
    assert CARD_VERDICT["borderline"] not in p1 and CARD_VERDICT["clinical"] not in p1


def test_curve_howto_block_appears_once_on_page_one():
    html = build_pending_report_html(_profile("p2_partial_borderline"))
    assert html.count(CURVE_HOWTO_CAPTION) == 1
    for line in CURVE_HOWTO_LINES:
        assert html.count(line) == 1
    assert html.index('id="howto"') < html.index('id="page2"')
    # 카드 곡선의 꼭짓점 라벨은 짧은 형태만 쓴다 (긴 문장은 설명 블록에만)
    assert html.count("또래가 가장 많이 모여 있는 점수대") == 1
    # 카드 곡선마다 하나 (접힌 종합지표 2개는 곡선을 그리지 않으므로 전체 svg 수 - 렌즈 2개)
    assert html.count("또래 평균 50T") == html.count("<svg") - 2 == 9


def test_card_fixed_texts_stay_outside_llm_gates():
    """카드 고정 문구는 템플릿에서만 나오고 가드레일·품질 지표가 보는 LLM 텍스트에는 없다.

    "높은 편"은 G8이 LLM 출력에서 금지하는 비표준 밴드 표현이지만, 카드의 결론 문장은
    같은 줄의 원 보고서 라벨 배지와 항상 짝인 고정 문구라 게이트 대상이 아니다.
    """
    profile = _profile("p2_partial_borderline")
    results = generate_all(profile, make_client("mock"))
    html = build_report_html(profile, results)
    fixed = [*CARD_VERDICT.values(), *CARD_PLAIN_RANGE.values(), CARD_NOT_DIAGNOSIS]
    assert all(text in html for text in fixed if text != CARD_VERDICT["clinical"]
               and text != CARD_PLAIN_RANGE["clinical"])
    gated = " ".join(t for task in ("explain", "prep")
                     for _b, t in caregiver_texts(task, results[task].output))
    assert all(text not in gated for text in fixed)
    for task in ("explain", "prep"):
        assert check_output(profile, task, results[task].output) == []


def test_normal_cards_have_single_line_header():
    """정상 카드(접힘)는 헤더 한 줄: 척도명, 결론, 라벨 배지, T점수. 둘째 줄(쉬운 구간 문장)은 준임상 이상에만."""
    html = build_pending_report_html(_profile("p2_partial_borderline"))
    somatic = _card(html, "somatic")
    assert CARD_VERDICT["normal"] in somatic
    assert '<span class="chip normal">정상</span>' in somatic and "51T" in somatic
    assert "전문용어로" not in somatic and '<p class="range">' not in somatic
    internalizing = _card(html, "internalizing")
    assert "전문용어로" in internalizing and CARD_PLAIN_RANGE["borderline"] in internalizing
    p1 = build_pending_report_html(_profile("p1_all_normal"))
    assert "전문용어로" not in p1
