"""종형곡선 렌더러 테스트: 기준선 라벨 겹침 회피."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser import BandCriteria
from src.renderer import bell_curve_svg


def _label_ys(svg: str) -> dict[str, float]:
    """기준선 라벨 텍스트 → y 좌표."""
    return {m.group(2): float(m.group(1))
            for m in re.finditer(r'<text x="[\d.]+" y="([\d.]+)" font-size="9.5"[^>]*>([^<]*기준 \d+T)</text>', svg)}


def test_composite_boundary_labels_are_staggered():
    """종합지표(준임상 60T·임상 63T, 3T 간격)는 두 번째 라벨이 한 줄 아래로 내려간다."""
    ys = _label_ys(bell_curve_svg(62, BandCriteria(normal_max_t=59, borderline_max_t=62)))
    assert set(ys) == {"준임상 기준 60T", "임상 기준 63T"}
    assert ys["임상 기준 63T"] > ys["준임상 기준 60T"]


def test_syndrome_boundary_labels_stay_on_one_line():
    """개별 척도(60T/70T)는 영향 없이 같은 줄에 남는다."""
    ys = _label_ys(bell_curve_svg(67, BandCriteria(normal_max_t=59, borderline_max_t=69)))
    assert ys["임상 기준 70T"] == ys["준임상 기준 60T"]


# ---------------------------------------------------------------- 카드 곡선 개편 (ADR 0008)

from src import renderer as R  # noqa: E402
from src.renderer import concept_curve_svg, curve_explainer_svg  # noqa: E402

COMP = BandCriteria(normal_max_t=59, borderline_max_t=62)
SYN = BandCriteria(normal_max_t=59, borderline_max_t=69)
_TEXT = re.compile(r'<text x="([\d.]+)" y="([\d.-]+)"(?: text-anchor="(\w+)")? font-size="([\d.]+)"[^>]*>([^<]*)</text>')


def _plot_label_boxes(svg: str, base: float) -> dict[str, tuple]:
    """기준선 위(플롯 영역) 라벨의 추정 상자: 꼭짓점, 기준선 라벨 2개, 마커 라벨, 오차 범위 라벨."""
    boxes = {}
    for m in _TEXT.finditer(svg):
        x, y, anchor, fs, text = float(m.group(1)), float(m.group(2)), m.group(3) or "start", float(m.group(4)), m.group(5)
        if y < base:
            boxes[text] = R._text_box(x, y, anchor, text, fs)
    return boxes


def test_card_curve_has_new_visual_language():
    svg = bell_curve_svg(62, COMP, clip_id="clip-x")
    assert '<clipPath id="clip-x">' in svg and 'clip-path="url(#clip-x)"' in svg
    assert svg.count("<rect") >= 3 + 3          # 구간 3 + 이름표 견본 3
    assert "fill-opacity" not in svg            # 반투명 SEM 사각형은 더 이상 없다
    assert "또래 평균 50T" in svg and "이번 결과 62T" in svg and "우리 아이" not in svg
    assert "오차 범위 58~66T (예시값)" in svg
    for label in ("평균 범위", "함께 살펴볼 범위", "임상 기준 이상"):
        assert label in svg
    # 렌즈 곡선(구간 색 없음)과 설명용 곡선(마커 없음)
    assert "clipPath" not in concept_curve_svg()
    ex = curve_explainer_svg()
    assert "또래가 가장 많이 모여 있는 점수대" in ex and "오른쪽일수록" in ex
    assert "이번 결과" not in ex and "오차 범위" not in ex and "준임상" not in ex


def test_plot_labels_never_overflow_overlap_or_cross_the_curve():
    """25~85T 전부, 두 기준표 모두: 플롯 영역 라벨은 화면 안에 있고 서로 겹치지 않으며
    마커·오차 범위 라벨은 곡선을 가로지르지 않는다 (렌더러의 폭 추정치 기준)."""
    width = 460
    x, y, base, t_at = R._geometry(width, R.CARD_HEIGHT, top=R.CARD_TOP, bottom=R.CARD_BOTTOM)
    for criteria in (COMP, SYN):
        for t in range(25, 86):
            svg = bell_curve_svg(t, criteria)
            boxes = _plot_label_boxes(svg, base)
            assert len(boxes) == 5, (t, sorted(boxes))
            items = list(boxes.items())
            for text, box in items:
                assert box[0] >= 0 and box[2] <= width, (t, criteria, text, box)
            for i, (ta, a) in enumerate(items):
                for tb, b in items[i + 1:]:
                    assert not R._overlaps(a, b, pad=0.0), (t, criteria, ta, tb)
            for text in (f"이번 결과 {t}T", f"오차 범위 {t - 4}~{t + 4}T (예시값)"):
                assert not R._crosses_curve(boxes[text], y, t_at), (t, criteria, text)
