"""2페이지 정적 HTML 리포트 생성.

1페이지 '관찰자의 렌즈'(고정 문구 + 개념 예시 곡선 + '곡선 읽는 법' 블록),
2페이지 '우리 아이 결과'(총점 위계 + 척도 카드), 이어서 상담 준비 도우미
섹션. 서버 없이 브라우저로 열면 끝나는 단일 파일이며 외부 CDN을 쓰지 않는다.

척도 카드의 정보 순서는 고정이다 (ADR 0008): 척도명과 한 문장 결론, 쉬운
구간 이름 + 원 보고서 라벨 배지 + 수치, 종형곡선, 척도별 고정 해설, 그리고
준임상·임상 카드에만 "이 점수만으로 진단하지 않아요" 한 줄. 결론과 구간
이름은 밴드별 고정 문구라 LLM도 가드레일도 거치지 않는다.

LLM 문장은 네 자리뿐이다: 전체 요약(보호자 관찰과 소견을 잇는 연결
문단), 질문, 관찰 포인트, 상담사용 요약. 상담 전 안내(PRE_COUNSELING_NOTE,
ADR 0009), 척도 카드 본문과 심리교육 문단(렌즈 안내, T점수 설명, 준임상
일반론, 한계 고지)은 전부 사전 작성 고정 문구다 (scale_texts.py) - 일반론과
개별 단정의 거리가 한 문장이라 LLM에 맡기지 않고, 고정 문구는 검증도 필요 없다.

길이 원칙: 사람들은 긴 글을 읽지 않는다. 정상 범위 척도는 접힌 한 줄,
준임상·임상 척도만 펼친 카드, 문단은 3문장 상한, 각주·메타는 접기,
첫 화면에서 상담 준비 섹션으로 바로 가는 앵커.
"""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .guardrails import SafeResult
from .parser import BAND_KO, COMPOSITE_IDS, SYNDROME_IDS, CBCLProfile, SCALE_NAMES
from .renderer import (DEFAULT_SEM, EXAMPLE_RELIABILITY, bell_curve_svg, concept_curve_svg,
                       curve_explainer_svg)
from .scale_texts import LIMITS_TEXT, scale_card_text

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# ---- 고정 문구 (사전 작성, LLM 미사용) ----
LENS_QUOTE = (
    "이 결과는 보호자의 눈으로 본 아이입니다. 관찰자가 다르면(선생님, 아이 자신) "
    "결과가 다르게 나오는 것이 정상이고, 그 차이 자체가 중요한 정보입니다. "
    "그래서 상담사와의 대화에서 그림이 완성됩니다."
)
TSCORE_EXPLAIN = (
    "T점수는 또래 평균을 50, 표준편차를 10으로 맞춘 점수입니다. "
    "또래 100명을 줄 세우면 50은 한가운데, 60은 위에서 16번째쯤, 70은 위에서 2~3번째쯤입니다. "
    "점수는 점이 아니라 구간으로 읽는 것이 정확하며, 2페이지 곡선 위에 굵게 표시한 오차 구간이 그 범위입니다."
)
FOOTNOTES_SUMMARY = "보호자와 교사의 평가 상관은 평균 r=.28 (메타분석 2건) - 불일치는 오류가 아니라 정보입니다."
FOOTNOTES = [
    "서로 다른 위치의 관찰자(부모·교사)가 같은 아동을 평가할 때 상관은 평균 r=.28입니다. "
    "119개 연구 메타분석의 수치이고(Achenbach, McConaughy & Howell, 1987), 341개 연구를 "
    "다시 모은 후속 메타분석에서도 전체 평균 r=.28로 재확인되었습니다(De Los Reyes et al., 2015).",
    "한국 임상 표본(6~12세 임상의뢰 아동 165명)에서도 K-CBCL 부모-교사 평정의 내재화 상관은 "
    ".18이었고, 우울/불안 척도는 유의한 상관이 없었습니다(김흥규·안정숙·김민혁, 2012).",
    "이 불일치는 측정 오류가 아니라 정보입니다. 아동의 행동이 집과 학교에서 실제로 다르고 "
    "관찰자마다 보이는 부분이 다르기 때문이며, 맥락 차이를 임상적으로 의미 있는 정보로 "
    "받아들이라는 것이 현대 임상 문헌의 권고입니다(De Los Reyes & Kazdin, 2005; Dirks et al., 2012).",
]
SEM_NOTE = (
    f"곡선 아래 색은 원 보고서 기준표의 구간(정상, 준임상, 임상)이고, 점선은 그 기준선입니다. "
    f"마커 주변에서 굵게 표시한 곡선 구간은 측정의 표준오차(SEM)를 나타냅니다. "
    f"예시 신뢰도({str(EXAMPLE_RELIABILITY).lstrip('0')})로 계산한 ±1 표준오차 범위이며, SEM은 {DEFAULT_SEM}T입니다. "
    f"실제 서비스는 검사 매뉴얼의 척도별 신뢰도 계수를 씁니다."
)
# 1페이지 '곡선 읽는 법' 블록 (곡선의 의미는 여기서 한 번만 가르치고 카드마다 반복하지 않는다)
CURVE_HOWTO_CAPTION = "곡선이 높을수록 이 점수대에 해당하는 또래가 많아요"
CURVE_HOWTO_LINES = (
    "가운데 50T 근처에 또래의 대부분이 있고, 오른쪽으로 갈수록 그렇게 보고된 아이가 드물어집니다.",
    "2페이지의 곡선마다 같은 그림 위에 이번 결과의 위치와 오차 범위를 표시합니다.",
)
# 척도 카드 상단의 밴드별 고정 문구 (ADR 0008). 어미는 관찰자 프레임("보고됐어요")을 유지하고
# 심각성 단정과 완화를 모두 피한다. 쉬운 구간 이름은 같은 줄의 원 보고서 라벨 배지와 항상 짝이다.
CARD_VERDICT = {
    # 결론 줄은 평가 형용사("높은 편") 없이 원 보고서 라벨을 풀어 쓴 구간 이름만 쓴다.
    # "선별 관찰 요망"(준임상)을 "상담에서 함께 살펴볼 범위"로 옮긴 것이지 새 판정이 아니다. G8이 LLM에
    # 금지하는 어휘를 시스템도 쓰지 않는다.
    "normal": "또래 평균 범위",
    "borderline": "상담에서 함께 살펴볼 범위",
    "clinical": "상담에서 우선 살펴볼 범위",
}
CARD_PLAIN_RANGE = {
    "normal": "또래 평균 범위",
    "borderline": "상담에서 함께 살펴볼 범위",
    "clinical": "상담에서 우선 살펴볼 범위",
}
# 준임상·임상 카드의 해설 아래 한 줄 (정상 카드에는 넣지 않는다)
CARD_NOT_DIAGNOSIS = "이 점수만으로 진단하지 않아요 · 상담에서 어떤 상황에서 나타났는지 함께 확인해요"
BORDERLINE_NOTE = (
    "준임상 구간은 확정된 상태가 아니라 관찰과 개입의 여지가 있는 구간입니다. "
    "해석 유의사항은 재검사와 다중 정보원(교사 보고, 자기 보고) 병행을 권고합니다."
)
CAUTION = ("이 보고서는 선별 도구이며 진단이 아닙니다. 검사 한 번의 결과는 "
           "아이를 이해하는 출발점일 뿐, 그 자체로 어떤 판정도 확정하지 않습니다.")
# 상담 준비 섹션 첫 줄의 고정 문구 (ADR 0009). 원래 LLM 블록(before_counseling)이던 자리인데,
# 실측 6런에서 모델 간 차이가 없었고 조언("이렇게 해 보세요")과 판정으로 흐를 위험이 가장 큰
# 자리라 고정 문구로 내렸다. 감정 인정 한 문장과 "수치의 의미는 상담에서" 한 문장뿐이며,
# LLM 출력이 아니므로 가드레일도 품질 지표도 거치지 않는다. 라벨에 "마음가짐"이라는 말은 쓰지 않는다.
PRE_COUNSELING_LABEL = "상담 전 안내"
PRE_COUNSELING_NOTE = (
    "검사 결과를 보고 걱정되는 마음이 드는 것은 자연스럽습니다. "
    "수치의 의미는 예약된 상담에서 상담사와 함께 확인하게 됩니다."
)
# LLM 생성 자리의 자리표시 문구 (탐색 콘솔의 생성 전 미리보기용, 리포트 산출물에는 나가지 않음)
PENDING_TEXT = "생성 대기 - '리포트 생성'을 누르면 보호자 문장을 인용한 생성 문구가 이 자리에 들어옵니다."


def _template_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(enabled_extensions=("html", "j2")),
    )


def _schedule_aware_text(text: str, counseling_scheduled: bool) -> str:
    """미예약 상태에서는 남은 날짜와 예약 사실을 전제하지 않는 문구로 바꾼다."""
    if counseling_scheduled:
        return text
    text = re.sub(r"상담까지\s*남은\s*\d+일\s*동안", "상담 예약 후 상담 전까지", text)
    text = re.sub(r"상담까지\s*\d+일\s*남은\s*시점의", "상담 예약 후 사용할", text)
    text = re.sub(r"상담까지\s*남은\s*\d+일", "상담 예약 후", text)
    text = re.sub(r"상담까지\s*\d+일", "상담 예약 후", text)
    return text.replace("예약된 상담에서", "상담 예약 후").replace("예약된 상담", "상담 예약 후")


def _scale_view(profile: CBCLProfile, sid: str) -> dict:
    """템플릿에 넘길 척도 카드 1개 분량의 데이터 (본문은 고정 문구)."""
    scale = profile.scale_map()[sid]
    return {
        "id": sid,
        "name": scale.name_ko,
        "t": scale.t_score,
        "band": scale.band,
        "band_ko": BAND_KO[scale.band],
        "open": scale.band != "normal",      # 정상 범위는 접힌 카드 (헤더 두 줄만 보임)
        "verdict": CARD_VERDICT[scale.band],
        "plain_range": CARD_PLAIN_RANGE[scale.band],
        "svg": bell_curve_svg(scale.t_score, profile.criteria_for(sid), clip_id=f"clip-{sid}"),
        "text": _schedule_aware_text(
            scale_card_text(sid, scale.band), profile.counseling_scheduled
        ),
    }


def _items_view(profile: CBCLProfile, items: list, text_key: str) -> list[dict]:
    names = SCALE_NAMES
    return [{"text": _schedule_aware_text(
                 it.get(text_key, ""), profile.counseling_scheduled
             ),
             "source_name": names.get(it.get("source_scale"), "원 보고서"),
             "fallback": bool(it.get("_fallback")),
             "pending": bool(it.get("_pending"))}
            for it in items if isinstance(it, dict)]


def build_crisis_html(profile: CBCLProfile) -> str:
    """위기 신호 검출 시의 전용 화면: 상담 연결 안내 + 즉시 도움 라인.

    해설도 수치도 넣지 않는다. 검출된 키워드도 다시 보여주지 않는다
    (화면의 역할은 해설이 아니라 연결이다).
    """
    return _template_env().get_template("crisis.html.j2").render(
        alias=profile.child.alias,
        instrument=profile.instrument,
        test_date=profile.test_date,
        counseling_scheduled=profile.counseling_scheduled,
        days=profile.days_until_counseling,
    )


def pending_results(profile: CBCLProfile) -> dict[str, SafeResult]:
    """LLM 결과가 아직 없을 때 결정론 부분만 미리 보기 위한 자리표시 SafeResult 2건.

    탐색 콘솔이 슬라이더 변경마다 곡선과 오차 구간 및 고정 문구를 즉시 다시 그리는 데 쓴다.
    생성 블록 4개는 전부 PENDING_TEXT이고 _pending 표식으로 템플릿이 구분한다.
    """
    return {
        "explain": SafeResult(task="explain", output={"overview": PENDING_TEXT}),
        "prep": SafeResult(task="prep", output={
            "questions_for_counselor": [{"question": PENDING_TEXT, "source_scale": None, "_pending": True}],
            "observation_points": [{"point": PENDING_TEXT, "source_scale": None, "_pending": True}],
            "counselor_briefing": PENDING_TEXT}),
    }


def build_pending_report_html(profile: CBCLProfile, mode_label: str = "생성 전") -> str:
    """생성 전 미리보기: 같은 템플릿·같은 렌더러로 결정론 부분만 실제 값으로 그린다."""
    return build_report_html(profile, pending_results(profile), mode_label=mode_label, pending=True)


def build_report_html(profile: CBCLProfile, results: dict, mode_label: str = "mock",
                      model_label: str = "", pending: bool = False) -> str:
    """SafeResult 2건({"explain","prep"})을 받아 완성 HTML 문자열을 만든다.

    pending=True면 생성 블록 자리에 '생성 대기' 표식을 붙인다 (탐색 콘솔 미리보기).
    """
    explain, prep = results["explain"], results["prep"]
    fallback_blocks = set(explain.fallback_blocks) | set(prep.fallback_blocks)
    elevated = [s.name_ko for s in profile.elevated_scales()]
    return _template_env().get_template("report.html.j2").render(
        alias=profile.child.alias,
        instrument=profile.instrument,
        test_date=profile.test_date,
        mode_label=mode_label,
        model_label=model_label,
        lens_quote=LENS_QUOTE,
        tscore_explain=TSCORE_EXPLAIN,
        footnotes_summary=FOOTNOTES_SUMMARY,
        footnotes=FOOTNOTES,
        concept_svg=concept_curve_svg(),
        explainer_svg=curve_explainer_svg(),
        curve_howto_caption=CURVE_HOWTO_CAPTION,
        curve_howto_lines=CURVE_HOWTO_LINES,
        not_diagnosis=CARD_NOT_DIAGNOSIS,
        sem_note=SEM_NOTE,
        borderline_note=BORDERLINE_NOTE,
        caution=CAUTION,
        limits=_schedule_aware_text(LIMITS_TEXT, profile.counseling_scheduled),
        elevated_names=elevated,
        counseling_scheduled=profile.counseling_scheduled,
        special_scales_administered=profile.special_scales_administered,
        has_borderline=any(s.band == "borderline" for s in profile.all_scales()),
        overview=_schedule_aware_text(
            explain.output["overview"], profile.counseling_scheduled
        ),
        overview_fallback="overview" in fallback_blocks,
        composites=[_scale_view(profile, sid) for sid in COMPOSITE_IDS],
        syndromes=[_scale_view(profile, sid) for sid in SYNDROME_IDS],
        pre_counseling_label=PRE_COUNSELING_LABEL,
        pre_counseling_note=_schedule_aware_text(
            PRE_COUNSELING_NOTE, profile.counseling_scheduled
        ),
        questions=_items_view(profile, prep.output["questions_for_counselor"], "question"),
        observations=_items_view(profile, prep.output["observation_points"], "point"),
        briefing=_schedule_aware_text(
            prep.output["counselor_briefing"], profile.counseling_scheduled
        ),
        briefing_fallback="counselor_briefing" in fallback_blocks,
        days=profile.days_until_counseling,
        regen_count=explain.regen_count + prep.regen_count,
        fallback_count=len(fallback_blocks),
        pending=pending,
    )
