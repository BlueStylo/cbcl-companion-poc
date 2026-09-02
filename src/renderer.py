"""종형곡선 SVG 렌더러 (결정론, matplotlib 미사용).

정규분포 곡선 아래 면적을 구간 색(정상/준임상/임상)으로 나누고, 아이 위치
마커와 SEM 대칭 오차 구간(곡선을 따라 굵게), 기준선, 구간 이름표를 그린다. 그래픽은
텍스트보다 판정처럼 읽히므로 전부 숫자에서 기계적으로 생성하고 확률
모델은 개입하지 않는다.

시각 언어는 두 겹이다. 구간 색은 곡선 아래에만 칠해 "또래 분포의 어느
부분인가"를 보여 주고, 아이의 결과는 마커(점, 수직 스템)와 좌우 대칭
곡선을 따라가는 굵은 오차 구간(SEM ±)으로 그 위에 얹는다. 두 겹이 같은 자리에서 반투명으로
섞이지 않도록 오차 범위는 면이 아니라 선이다.

라벨 배치는 후보 위치를 순서대로 시도해 (1) 화면 밖으로 나가지 않고
(2) 곡선을 가로지르지 않으며 (3) 고정 라벨·먼저 놓인 라벨과 겹치지 않는
첫 후보를 고른다. 글자 폭은 글꼴별 실측 대신 보수적 추정치(한글 1em)를
쓴다.
"""

from __future__ import annotations

import math

from .parser import BandCriteria

# 축 라벨은 렌즈 프레임과 일관되게 "많이 있음"이 아니라 "많이 보고됨"으로 고정
AXIS_LABEL = "오른쪽일수록 해당 행동이 또래보다 많이 보고됨 →"
# 꼭짓점 라벨 (카드는 좁으므로 짧게, 긴 설명은 1페이지 '곡선 읽는 법' 블록에만)
PEAK_LABEL = "또래 평균 50T"
EXPLAINER_PEAK_LABEL = "또래가 가장 많이 모여 있는 점수대"
# 구간 이름표 (쉬운 말). 색 경계는 기준표 그대로이고 이름표만 오해를 줄인다
ZONE_LABELS = {"normal": "평균 범위", "borderline": "함께 살펴볼 범위", "clinical": "임상 기준 이상"}
SEM_LABEL = "오차 범위 {lo}~{hi}T (예시값)"
DEFAULT_MARKER_LABEL = "이번 결과"

T_MEAN, T_SD = 50.0, 10.0
T_LEFT, T_RIGHT = 25.0, 85.0

# 신뢰도 예시값 .84 기준: SEM = 10 x sqrt(1 - 0.84) = 4.0T (예시값임을 화면에 표기)
EXAMPLE_RELIABILITY = 0.84
DEFAULT_SEM = round(T_SD * math.sqrt(1.0 - EXAMPLE_RELIABILITY), 1)

# 기준선 라벨("준임상 기준 60T", 9.5px 글꼴로 약 70px)이 두 기준선 사이에 들어가지
# 못할 만큼 좁으면 두 번째 라벨을 한 줄 아래로 어긋나게 놓는다. 종합지표(준임상 60T,
# 임상 63T: 3T = 약 22px)가 해당하고, 개별 척도(60T/70T: 약 72px)는 그대로 한 줄이다.
MIN_LABEL_GAP_PX = 60
LABEL_LINE_HEIGHT = 12

# 카드 곡선의 세로 여백. 위쪽은 꼭짓점 라벨과 꼭짓점 근처 마커 라벨이 겹치지 않을 만큼,
# 아래쪽은 눈금 라벨, 구간 이름표 줄, 축 라벨 세 줄 분량이다.
CARD_TOP, CARD_BOTTOM = 34, 48
CARD_HEIGHT = 168

_COL = {
    "curve": "#4a6fa5",
    "baseline": "#8a94a3",
    "normal_zone": "#f1f3f6",     # 곡선 아래 정상 구간 (아주 옅은 회청색)
    "border_zone": "#f0e6d8",     # 곡선 아래 준임상 구간 (채도 낮춤)
    "clinical_zone": "#ecdad3",   # 곡선 아래 임상 구간 (채도 낮춤)
    "boundary": "#c2ae92",
    "marker": "#b0552e",
    "text": "#4b5563",
    "faint": "#6b7280",
}


def _pdf01(t: float) -> float:
    """평균 50, 표준편차 10 정규곡선. 꼭짓점을 1로 정규화한 높이."""
    return math.exp(-((t - T_MEAN) ** 2) / (2 * T_SD ** 2))


def _geometry(width: int, height: int, top: int = 24, bottom: int = 40):
    """t값 <-> SVG 좌표 변환 함수와 기준선 좌표를 만든다."""
    pad_l, pad_r = 14, 14
    baseline_y = height - bottom
    amp = baseline_y - top
    span = (width - pad_l - pad_r) / (T_RIGHT - T_LEFT)

    def x(t: float) -> float:
        return pad_l + (t - T_LEFT) * span

    def y(t: float) -> float:
        return baseline_y - amp * _pdf01(t)

    def t_at(px: float) -> float:
        return T_LEFT + (px - pad_l) / span

    return x, y, baseline_y, t_at


def _curve_path(x, y) -> str:
    pts = []
    t = T_LEFT
    while t <= T_RIGHT + 1e-9:
        pts.append(f"{x(t):.1f},{y(t):.1f}")
        t += 1.0
    return "M" + " L".join(pts)


def _curve_segment_path(x, y, t_lo: float, t_hi: float, step: float = 0.5) -> str:
    """t_lo~t_hi 구간의 곡선 경로 (오차 구간을 곡선 위에 굵게 덧그릴 때 사용)."""
    pts = []
    t = t_lo
    while t < t_hi - 1e-9:
        pts.append(f"{x(t):.1f},{y(t):.1f}")
        t += step
    pts.append(f"{x(t_hi):.1f},{y(t_hi):.1f}")
    return "M" + " L".join(pts)


def _perp_cap(x, y, t: float, half_len: float = 5.0, dt: float = 0.25):
    """t 지점에서 곡선에 수직인 짧은 캡의 양 끝 좌표 (접선은 좌우 dt 차분으로 근사)."""
    x0, y0 = x(t), y(t)
    dx, dy = x(t + dt) - x(t - dt), y(t + dt) - y(t - dt)
    norm = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / norm, dx / norm
    return (x0 - nx * half_len, y0 - ny * half_len, x0 + nx * half_len, y0 + ny * half_len)


def _area_path(x, y, base: float) -> str:
    """곡선과 기준선으로 닫은 면적 (구간 색 클리핑용)."""
    return f"{_curve_path(x, y)} L{x(T_RIGHT):.1f},{base} L{x(T_LEFT):.1f},{base} Z"


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


# ---------------------------------------------------------------- 라벨 배치

def _est_width(text: str, font_size: float) -> float:
    """글자 폭 보수 추정 (한글 1em, 영숫자 0.55em, 공백 0.3em, 괄호 0.35em)."""
    w = 0.0
    for ch in text:
        if "가" <= ch <= "힣":
            w += 1.0
        elif ch == " ":
            w += 0.3
        elif ch in "()":
            w += 0.35
        elif ch.isascii():
            w += 0.55
        else:
            w += 0.8
    return w * font_size


def _text_box(ax: float, baseline: float, anchor: str, text: str, font_size: float):
    """(x1, y1, x2, y2). 한글 글꼴은 em 높이를 거의 다 쓰므로 위 0.85em, 아래 0.2em."""
    w = _est_width(text, font_size)
    x1 = {"start": ax, "end": ax - w, "middle": ax - w / 2}[anchor]
    return (x1, baseline - 0.85 * font_size, x1 + w, baseline + 0.2 * font_size)


def _overlaps(a, b, pad: float = 1.0) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def _crosses_curve(box, y, t_at, step: float = 2.0) -> bool:
    """상자 가로 구간 안에서 곡선이 상자의 세로 구간을 지나는지 (2px 간격 표본)."""
    px = box[0]
    while px <= box[2]:
        cy = y(t_at(px))
        if box[1] - 1.0 <= cy <= box[3] + 1.0:
            return True
        px += step
    return False


def _valid_placements(candidates, text, font_size, width, base, y, t_at, obstacles):
    """후보 (x, baseline, anchor) 가운데 화면 안이고 곡선을 가로지르지 않으며 장애물과
    겹치지 않는 것을 순서대로 낸다."""
    for ax, by, anchor in candidates:
        box = _text_box(ax, by, anchor, text, font_size)
        if box[0] < 2 or box[2] > width - 2 or box[1] < 2 or box[3] > base - 2:
            continue
        if _crosses_curve(box, y, t_at):
            continue
        if any(_overlaps(box, o) for o in obstacles):
            continue
        yield (ax, by, anchor), box


def _place_pair(first, second, width, base, y, t_at, obstacles):
    """라벨 두 개(마커 라벨, 오차 범위 라벨)를 함께 배치한다.

    first/second = (candidates, text, font_size). 첫 라벨의 유효 후보 순서대로 두 번째
    라벨이 들어갈 자리가 있는 첫 조합을 고른다 (첫 라벨만 욕심내면 꼬리 쪽 점수에서
    두 번째 라벨의 자리가 없어진다). 전부 실패하면 첫 후보들을 그대로 쓴다.
    """
    c1, t1, f1 = first
    c2, t2, f2 = second
    for p1, b1 in _valid_placements(c1, t1, f1, width, base, y, t_at, obstacles):
        for p2, _b2 in _valid_placements(c2, t2, f2, width, base, y, t_at, [*obstacles, b1]):
            return p1, p2
    for p1, _b1 in _valid_placements(c1, t1, f1, width, base, y, t_at, obstacles):
        # 두 번째 라벨의 유효 자리가 없으면(극단 꼬리) 겹침은 허용하되, 화면 좌우 안에 있고
        # 곡선을 가로지르지 않는 첫 후보를 고른다
        for ax, by, anchor in c2:
            box = _text_box(ax, by, anchor, t2, f2)
            if box[0] >= 2 and box[2] <= width - 2 and not _crosses_curve(box, y, t_at):
                return p1, (ax, by, anchor)
        return p1, c2[0]
    return c1[0], c2[0]


def _zone_legend_svg(x, base: float, b_start: float, c_start: float, width: int) -> str:
    """축 아래 구간 이름표 줄: 색 견본 + 이름을 각 구간 가운데 아래에 둔다.

    곡선 아래 면적이 좁아지는 오른쪽 꼬리 안에는 글자가 들어가지 않으므로 축 아래에
    둔다. 이웃과 겹치면 오른쪽으로 밀고, 화면 오른쪽 끝을 넘지 않게 한다.
    """
    row_y = base + 28
    fs, sw, gap = 9, 7, 4
    items = (
        ("normal", (T_LEFT + b_start) / 2, _COL["normal_zone"]),
        ("borderline", (b_start + c_start) / 2, _COL["border_zone"]),
        ("clinical", (c_start + T_RIGHT) / 2, _COL["clinical_zone"]),
    )
    parts, prev_right = [], -1e9
    for band, t_mid, color in items:
        label = ZONE_LABELS[band]
        w = sw + gap + _est_width(label, fs)
        x1 = max(x(t_mid) - w / 2, prev_right + 8)
        x1 = min(x1, width - 2 - w)
        parts.append(
            f'<rect x="{x1:.1f}" y="{row_y - sw:.1f}" width="{sw}" height="{sw}" fill="{color}" '
            f'stroke="{_COL["boundary"]}" stroke-width="0.6"/>'
            f'<text x="{x1 + sw + gap:.1f}" y="{row_y}" font-size="{fs}" fill="{_COL["faint"]}">{label}</text>'
        )
        prev_right = x1 + w
    return "".join(parts)


def _fmt_t(v: float) -> str:
    return f"{v:g}"


# ---------------------------------------------------------------- 카드 곡선

def bell_curve_svg(
    t_score: int,
    criteria: BandCriteria,
    sem: float = DEFAULT_SEM,
    width: int = 460,
    height: int = CARD_HEIGHT,
    marker_label: str = DEFAULT_MARKER_LABEL,
    clip_id: str | None = None,
) -> str:
    """척도 1개의 종형곡선 카드 그래픽.

    구성: 곡선 아래 면적을 정상/준임상/임상 구간 색으로 분할(clipPath), 정규곡선,
    경계 기준선(점선)과 기준선 라벨(곡선 위 빈 공간), 꼭짓점 라벨, 아이 위치 마커
    (점 + 수직 스템)와 곡선을 따라가는 SEM 대칭 오차 구간(양 끝 수직 캡) + 오차 범위 라벨, 눈금,
    축 아래 구간 이름표 줄, 축 라벨.

    clip_id는 문서 안에 곡선이 여러 개 들어갈 때 clipPath id 충돌을 피하기 위한
    접미사다 (같은 크기면 클립 형태는 동일하므로 충돌해도 그림은 같다).
    """
    x, y, base, t_at = _geometry(width, height, top=CARD_TOP, bottom=CARD_BOTTOM)
    top = CARD_TOP
    b_start = criteria.normal_max_t + 1       # 준임상 시작 (60T)
    c_start = criteria.borderline_max_t + 1   # 임상 시작 (70T 또는 63T)
    clip_id = clip_id or f"bell-clip-{width}x{height}"

    # 곡선 아래 면적만 구간 색으로 칠한다 (곡선 위 빈 공간은 라벨 자리)
    defs = f'<defs><clipPath id="{clip_id}"><path d="{_area_path(x, y, base)}"/></clipPath></defs>'
    zones = (
        f'<g clip-path="url(#{clip_id})">'
        f'<rect x="{x(T_LEFT):.1f}" y="{top}" width="{x(b_start) - x(T_LEFT):.1f}" '
        f'height="{base - top}" fill="{_COL["normal_zone"]}"/>'
        f'<rect x="{x(b_start):.1f}" y="{top}" width="{x(c_start) - x(b_start):.1f}" '
        f'height="{base - top}" fill="{_COL["border_zone"]}"/>'
        f'<rect x="{x(c_start):.1f}" y="{top}" width="{x(T_RIGHT) - x(c_start):.1f}" '
        f'height="{base - top}" fill="{_COL["clinical_zone"]}"/>'
        f"</g>"
    )

    # 경계 기준선 + 라벨 (곡선 위 빈 공간, 좁으면 두 번째 라벨을 한 줄 아래로)
    stagger = (x(c_start) - x(b_start)) < MIN_LABEL_GAP_PX
    cutoff_labels = []
    boundaries = []
    for i, (t, label) in enumerate(((b_start, "준임상 기준"), (c_start, "임상 기준"))):
        ly = top + 10 + (LABEL_LINE_HEIGHT if stagger and i else 0)
        text = f"{label} {t}T"
        boundaries.append(
            f'<line x1="{x(t):.1f}" y1="{top}" x2="{x(t):.1f}" y2="{base}" '
            f'stroke="{_COL["boundary"]}" stroke-width="1" stroke-dasharray="3,3"/>'
            f'<text x="{x(t) + 3:.1f}" y="{ly}" font-size="9.5" fill="{_COL["faint"]}">{text}</text>'
        )
        cutoff_labels.append(_text_box(x(t) + 3, ly, "start", text, 9.5))

    # 꼭짓점 라벨 (짧게)
    peak_y = top - 20
    peak = (
        f'<text x="{x(T_MEAN):.1f}" y="{peak_y}" text-anchor="middle" font-size="9.5" '
        f'fill="{_COL["faint"]}">{PEAK_LABEL}</text>'
    )
    obstacles = [_text_box(x(T_MEAN), peak_y, "middle", PEAK_LABEL, 9.5), *cutoff_labels]

    # 마커: 수직 스템 + 점 + SEM 대칭 오차 구간. 오차 구간은 가로선이 아니라 곡선을 따라
    # 굵게 덧그리고(58~66T 구간의 곡선 자체), 양 끝에 곡선에 수직인 짧은 캡을 세운다.
    # 구간 색(면)과 시각 언어가 다르고, "이 점수대의 곡선 위 어디쯤"이 그대로 읽힌다.
    plot_t = min(max(float(t_score), T_LEFT), T_RIGHT)
    mx, my = x(plot_t), y(plot_t)
    lo = min(max(t_score - sem, T_LEFT), T_RIGHT)
    hi = min(max(t_score + sem, T_LEFT), T_RIGHT)
    lx, hx = x(lo), x(hi)
    ly, hy = y(lo), y(hi)
    cap_l, cap_h = _perp_cap(x, y, lo), _perp_cap(x, y, hi)
    marker = (
        f'<line x1="{mx:.1f}" y1="{base}" x2="{mx:.1f}" y2="{my:.1f}" '
        f'stroke="{_COL["marker"]}" stroke-width="1.5"/>'
        f'<path d="{_curve_segment_path(x, y, lo, hi)}" fill="none" stroke="{_COL["marker"]}" '
        f'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<line x1="{cap_l[0]:.1f}" y1="{cap_l[1]:.1f}" x2="{cap_l[2]:.1f}" y2="{cap_l[3]:.1f}" '
        f'stroke="{_COL["marker"]}" stroke-width="2" stroke-linecap="round"/>'
        f'<line x1="{cap_h[0]:.1f}" y1="{cap_h[1]:.1f}" x2="{cap_h[2]:.1f}" y2="{cap_h[3]:.1f}" '
        f'stroke="{_COL["marker"]}" stroke-width="2" stroke-linecap="round"/>'
        f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="4.5" fill="{_COL["marker"]}" stroke="#ffffff" stroke-width="1.2"/>'
    )

    # 마커 라벨: 점의 오른쪽 위가 기본, 안 되면 왼쪽 위, 범위선 바깥, 점 위 가운데(꼬리),
    # 오른쪽 끝 정렬(꼬리 끝), 점 아래 순. 오차 범위 라벨: 오른쪽 캡 옆이 기본, 안 되면
    # 왼쪽 캡 옆, 범위선 위 가운데, 오른쪽 끝 정렬, 점 아래, 마커 라벨 위 순.
    outside_suffix = " (표시 범위 밖)" if not T_LEFT <= t_score <= T_RIGHT else ""
    m_text = f"{marker_label} {t_score}T{outside_suffix}"
    s_text = SEM_LABEL.format(lo=_fmt_t(t_score - sem), hi=_fmt_t(t_score + sem))
    # 점 위 가운데와 화면 끝 정렬 후보는 점이 꼬리에 있을 때만 쓴다 (가운데 근처 점수에서는
    # 점에서 먼 자리나 자리가 들쭉날쭉 바뀌는 것을 피한다). 오차 범위 라벨의 마지막 후보
    # (점 아래 두 번째 줄)는 마커 라벨이 점 아래로 내려간 경우(58~59T)의 자리다.
    near_right, near_left = mx > width - 120, mx < 120
    tail_m = ([(width - 4, top - 8, "end")] if t_score > T_RIGHT else []) \
        + ([(mx, my - 24, "middle")] if near_right or near_left else []) \
        + ([(width - 4, my - 24, "end")] if near_right else []) + ([(4, my - 24, "start")] if near_left else [])
    tail_s = ([(width - 4, my - 12, "end")] if near_right else []) + ([(4, my - 12, "start")] if near_left else [])
    (ax, by, anchor), (sx, sy, s_anchor) = _place_pair(
        ([(mx + 6, my - 8, "start"), (mx - 6, my - 8, "end"),
          (lx - 6, my - 8, "end"), (hx + 6, my - 8, "start"), *tail_m,
          (mx + 6, my + 16, "start"), (mx - 6, my + 16, "end")], m_text, 11),
        ([(hx + 7, hy + 3, "start"), (lx - 7, ly + 3, "end"),
          (hx + 7, hy - 10, "start"), (lx - 7, ly - 10, "end"),
          (hx + 5, my + 3, "start"), (lx - 5, my + 3, "end"),
          (hx + 5, my - 12, "start"), (lx - 5, my - 12, "end"),
          (mx, my - 12, "middle"), *tail_s,
          (mx + 6, my + 16, "start"), (mx - 6, my + 16, "end"),
          (mx + 6, my - 20, "start"), (mx - 6, my - 20, "end"),
          (mx - 6, my + 28, "end"), (mx + 6, my + 28, "start")], s_text, 9),
        width, base, y, t_at, obstacles)
    marker_text = (
        f'<text x="{ax:.1f}" y="{by:.1f}" text-anchor="{anchor}" font-size="11" '
        f'font-weight="600" fill="{_COL["marker"]}">{m_text}</text>'
    )
    sem_text = (
        f'<text x="{sx:.1f}" y="{sy:.1f}" text-anchor="{s_anchor}" font-size="9" '
        f'fill="{_COL["marker"]}">{s_text}</text>'
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">'
        f"{defs}{zones}"
        f'<path d="{_curve_path(x, y)}" fill="none" stroke="{_COL["curve"]}" stroke-width="2"/>'
        f'<line x1="{x(T_LEFT):.1f}" y1="{base}" x2="{x(T_RIGHT):.1f}" y2="{base}" '
        f'stroke="{_COL["baseline"]}" stroke-width="1"/>'
        f"{''.join(boundaries)}{peak}{_ticks_svg(x, base)}"
        f"{marker}{marker_text}{sem_text}"
        f"{_zone_legend_svg(x, base, b_start, c_start, width)}"
        f'<text x="{width / 2:.0f}" y="{height - 6}" text-anchor="middle" font-size="11" '
        f'fill="{_COL["text"]}">{AXIS_LABEL}</text>'
        f"</svg>"
    )


# ---------------------------------------------------------------- 1페이지 설명용 곡선

def curve_explainer_svg(width: int = 420, height: int = 170) -> str:
    """1페이지 '곡선 읽는 법' 블록용 곡선: 마커 없음, 구간 색 없음.

    곡선 아래를 아주 옅게 칠해 "면적 = 또래"가 보이게 하고, 꼭짓점 위에 짧은
    지시선과 긴 라벨을 둔다. 축 라벨은 카드와 같은 문장이다. viewBox를 카드 곡선보다
    좁게 잡아 모바일(375px)에서도 라벨이 읽히게 한다 (템플릿에서 최대 폭을 제한).
    """
    top, bottom = 40, 40
    x, y, base, _t_at = _geometry(width, height, top=top, bottom=bottom)
    apex_x, apex_y = x(T_MEAN), y(T_MEAN)
    label_y = 16
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">'
        f'<path d="{_area_path(x, y, base)}" fill="{_COL["normal_zone"]}" stroke="none"/>'
        f'<path d="{_curve_path(x, y)}" fill="none" stroke="{_COL["curve"]}" stroke-width="2"/>'
        f'<line x1="{x(T_LEFT):.1f}" y1="{base}" x2="{x(T_RIGHT):.1f}" y2="{base}" '
        f'stroke="{_COL["baseline"]}" stroke-width="1"/>'
        f'<line x1="{apex_x:.1f}" y1="{apex_y:.1f}" x2="{apex_x:.1f}" y2="{base}" '
        f'stroke="{_COL["boundary"]}" stroke-width="1" stroke-dasharray="3,3"/>'
        f'<line x1="{apex_x:.1f}" y1="{label_y + 5}" x2="{apex_x:.1f}" y2="{apex_y - 3:.1f}" '
        f'stroke="{_COL["text"]}" stroke-width="1"/>'
        f'<circle cx="{apex_x:.1f}" cy="{apex_y:.1f}" r="3" fill="{_COL["curve"]}"/>'
        f'<text x="{apex_x:.1f}" y="{label_y}" text-anchor="middle" font-size="13" '
        f'fill="{_COL["text"]}">{EXPLAINER_PEAK_LABEL}</text>'
        f"{_ticks_svg(x, base)}"
        f'<text x="{width / 2:.0f}" y="{height - 6}" text-anchor="middle" font-size="12" '
        f'fill="{_COL["text"]}">{AXIS_LABEL}</text>'
        f"</svg>"
    )


def concept_curve_svg(width: int = 560, height: int = 200) -> str:
    """1페이지 '관찰자의 렌즈'용 개념 예시 곡선.

    같은 아이를 두 관찰자가 평가했을 때 결과가 다를 수 있음을 마커 2개로
    보여준다. 두 마커는 평균 50을 중심으로 대칭(45T, 55T)으로 놓아 "보호자가
    더 높게 본다"는 방향성이 그림에 실리지 않게 한다. 실측값이 아닌 개념
    예시임을 그림 안에 명시한다. 구간 색이 없으므로 클리핑할 면적도 없다.
    """
    x, y, base, _t_at = _geometry(width, height)
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
