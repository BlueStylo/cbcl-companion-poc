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
