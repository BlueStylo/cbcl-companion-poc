"""종형곡선 SVG 렌더러 (결정론, matplotlib 미사용).

정규분포 곡선 위에 아이 위치 마커, SEM 대칭 밴드, 준임상/임상 구간의
옅은 배경을 그린다. 그래픽은 텍스트보다 판정처럼 읽히므로 전부 숫자에서
기계적으로 생성하고 확률 모델은 개입하지 않는다.
"""

from __future__ import annotations

import math

from .parser import BandCriteria

# 축 라벨은 렌즈 프레임과 일관되게 "많이 있음"이 아니라 "많이 보고됨"으로 고정
AXIS_LABEL = "오른쪽일수록 해당 행동이 또래보다 많이 보고됨 →"

T_MEAN, T_SD = 50.0, 10.0
T_LEFT, T_RIGHT = 25.0, 85.0

# 신뢰도 예시값 .84 기준: SEM = 10 x sqrt(1 - 0.84) = 4.0T (예시값임을 화면에 표기)
EXAMPLE_RELIABILITY = 0.84
DEFAULT_SEM = round(T_SD * math.sqrt(1.0 - EXAMPLE_RELIABILITY), 1)

_COL = {
    "curve": "#4a6fa5",
    "baseline": "#8a94a3",
    "border_zone": "#f6e8d4",   # 준임상 구간 배경
    "clinical_zone": "#f3dbd2", # 임상 구간 배경
    "boundary": "#c2ae92",
    "sem": "#aac4e2",
    "marker": "#b0552e",
    "text": "#4b5563",
    "faint": "#6b7280",
}


def _pdf01(t: float) -> float:
    """평균 50, 표준편차 10 정규곡선. 꼭짓점을 1로 정규화한 높이."""
    return math.exp(-((t - T_MEAN) ** 2) / (2 * T_SD ** 2))


def _geometry(width: int, height: int):
    """t값 → SVG 좌표 변환 함수와 기준선 좌표를 만든다."""
    pad_l, pad_r, top, bottom = 14, 14, 24, 40
    baseline_y = height - bottom
    amp = baseline_y - top

    def x(t: float) -> float:
        return pad_l + (t - T_LEFT) / (T_RIGHT - T_LEFT) * (width - pad_l - pad_r)

    def y(t: float) -> float:
        return baseline_y - amp * _pdf01(t)

    return x, y, baseline_y


def _curve_path(x, y) -> str:
    pts = []
    t = T_LEFT
    while t <= T_RIGHT + 1e-9:
        pts.append(f"{x(t):.1f},{y(t):.1f}")
        t += 1.0
    return "M" + " L".join(pts)


def _ticks_svg(x, baseline_y) -> str:
    parts = []
    for t in (30, 40, 50, 60, 70, 80):
        parts.append(
            f'<line x1="{x(t):.1f}" y1="{baseline_y}" x2="{x(t):.1f}" y2="{baseline_y + 4}" '
            f'stroke="{_COL["baseline"]}" stroke-width="1"/>'
            f'<text x="{x(t):.1f}" y="{baseline_y + 15}" text-anchor="middle" '
            f'font-size="10" fill="{_COL["faint"]}">{t}T</text>'
        )
    return "".join(parts)


def bell_curve_svg(
    t_score: int,
    criteria: BandCriteria,
    sem: float = DEFAULT_SEM,
    width: int = 460,
    height: int = 185,
    marker_label: str = "우리 아이",
) -> str:
    """척도 1개의 종형곡선 카드 그래픽.

    구성: 준임상/임상 구간 옅은 배경, SEM 대칭 밴드, 정규곡선,
    아이 위치 마커, 경계 기준선, 축 라벨.
    """
    x, y, base = _geometry(width, height)
    b_start = criteria.normal_max_t + 1       # 준임상 시작 (60T)
    c_start = criteria.borderline_max_t + 1   # 임상 시작 (70T 또는 63T)
    top = 24

    zones = (
        f'<rect x="{x(b_start):.1f}" y="{top}" width="{x(c_start) - x(b_start):.1f}" '
        f'height="{base - top}" fill="{_COL["border_zone"]}"/>'
        f'<rect x="{x(c_start):.1f}" y="{top}" width="{x(T_RIGHT) - x(c_start):.1f}" '
        f'height="{base - top}" fill="{_COL["clinical_zone"]}"/>'
    )

    lo, hi = max(T_LEFT, t_score - sem), min(T_RIGHT, t_score + sem)
    sem_band = (
        f'<rect x="{x(lo):.1f}" y="{top}" width="{x(hi) - x(lo):.1f}" height="{base - top}" '
        f'fill="{_COL["sem"]}" fill-opacity="0.35"/>'
    )

    boundaries = "".join(
        f'<line x1="{x(t):.1f}" y1="{top}" x2="{x(t):.1f}" y2="{base}" '
        f'stroke="{_COL["boundary"]}" stroke-width="1" stroke-dasharray="3,3"/>'
        f'<text x="{x(t) + 3:.1f}" y="{top + 10}" font-size="9.5" fill="{_COL["faint"]}">{label} {t}T</text>'
        for t, label in ((b_start, "준임상 기준"), (c_start, "임상 기준"))
    )

    mx, my = x(t_score), y(t_score)
    anchor = "end" if mx > width - 90 else "start"
    lx = mx - 6 if anchor == "end" else mx + 6
    marker = (
        f'<line x1="{mx:.1f}" y1="{base}" x2="{mx:.1f}" y2="{my:.1f}" '
        f'stroke="{_COL["marker"]}" stroke-width="1.5"/>'
        f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="4" fill="{_COL["marker"]}"/>'
        f'<text x="{lx:.1f}" y="{my - 8:.1f}" text-anchor="{anchor}" font-size="11" '
        f'font-weight="600" fill="{_COL["marker"]}">{marker_label} {t_score}T</text>'
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">'
        f"{zones}{sem_band}"
        f'<path d="{_curve_path(x, y)}" fill="none" stroke="{_COL["curve"]}" stroke-width="2"/>'
        f'<line x1="{x(T_LEFT):.1f}" y1="{base}" x2="{x(T_RIGHT):.1f}" y2="{base}" '
        f'stroke="{_COL["baseline"]}" stroke-width="1"/>'
        f"{boundaries}{_ticks_svg(x, base)}{marker}"
        f'<text x="{width / 2:.0f}" y="{height - 6}" text-anchor="middle" font-size="11" '
        f'fill="{_COL["text"]}">{AXIS_LABEL}</text>'
        f"</svg>"
    )


def concept_curve_svg(width: int = 560, height: int = 200) -> str:
    """1페이지 '관찰자의 렌즈'용 개념 예시 곡선.

    같은 아이를 두 관찰자가 평가했을 때 결과가 다를 수 있음을 마커 2개로
    보여준다. 두 마커는 평균 50을 중심으로 대칭(45T, 55T)으로 놓아 "보호자가
    더 높게 본다"는 방향성이 그림에 실리지 않게 한다. 실측값이 아닌 개념
    예시임을 그림 안에 명시한다.
    """
    x, y, base = _geometry(width, height)
    examples = (("관찰자 A (예시)", 45, _COL["curve"], "end"),
                ("관찰자 B (예시)", 55, _COL["marker"], "start"))
    markers = []
    for label, t, color, anchor in examples:
        mx, my = x(t), y(t)
        lx = mx - 6 if anchor == "end" else mx + 6
        markers.append(
            f'<line x1="{mx:.1f}" y1="{base}" x2="{mx:.1f}" y2="{my:.1f}" '
            f'stroke="{color}" stroke-width="1.5" stroke-dasharray="2,2"/>'
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="4" fill="{color}"/>'
            f'<text x="{lx:.1f}" y="{my - 8:.1f}" text-anchor="{anchor}" font-size="10.5" fill="{color}">{label}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">'
        f'<path d="{_curve_path(x, y)}" fill="none" stroke="{_COL["curve"]}" stroke-width="2"/>'
        f'<line x1="{x(T_LEFT):.1f}" y1="{base}" x2="{x(T_RIGHT):.1f}" y2="{base}" '
        f'stroke="{_COL["baseline"]}" stroke-width="1"/>'
        f"{''.join(markers)}{_ticks_svg(x, base)}"
        f'<text x="{x(50):.1f}" y="16" text-anchor="middle" font-size="10.5" '
        f'fill="{_COL["faint"]}">평균 50T · 또래 대부분은 40~60T 사이</text>'
        f'<text x="{width - 10}" y="{28}" text-anchor="end" font-size="10" '
        f'fill="{_COL["faint"]}">특정 아동의 실측값이 아닌 개념 예시입니다</text>'
        f'<text x="{width / 2:.0f}" y="{height - 6}" text-anchor="middle" font-size="11" '
        f'fill="{_COL["text"]}">{AXIS_LABEL}</text>'
        f"</svg>"
    )
