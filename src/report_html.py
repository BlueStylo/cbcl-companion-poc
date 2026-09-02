"""2페이지 정적 HTML 리포트 생성.

1페이지 '관찰자의 렌즈'(고정 문구 + 개념 예시 곡선 + '곡선 읽는 법' 블록),
2페이지 '우리 아이 결과'(총점 위계 + 척도 카드), 이어서 상담 준비 도우미
섹션. 서버 없이 브라우저로 열면 끝나는 단일 파일이며 외부 CDN을 쓰지 않는다.

척도 카드의 정보 순서는 고정이다 (ADR 0008): 척도명과 한 문장 결론, 쉬운
구간 이름 + 원 보고서 라벨 배지 + 수치, 종형곡선, 척도별 고정 해설, 그리고
준임상·임상 카드에만 "이 점수만으로 진단하지 않아요" 한 줄. 결론과 구간
이름은 밴드별 고정 문구라 LLM도 가드레일도 거치지 않는다.

LLM 문장은 한 자리뿐이다: 상담사에게 물어볼 질문 (ADR 0010과 그 보강).
연결 문단(보호자 관찰과 검사 소견), 가정 관찰 포인트, 상담사에게 전달할 요약은 이 모듈이
결정론으로 조립한다 (build_overview_text, build_observation_points, build_counselor_briefing):
보호자 의견을 원문 그대로 큰따옴표로 인용하고, 상승 척도는 보고서 라벨 그대로 나열하며,
관찰 포인트는 준임상 이상 척도를 위계 순서로 골라 척도별 고정 문구를 붙인다.
LLM이 하던 "의견과 척도의 연결"은 질문의 근거 척도 배지가 대신한다. 상담 전 안내
(PRE_COUNSELING_NOTE, ADR 0009), 척도 카드 본문과 심리교육 문단(렌즈 안내, T점수 설명,
준임상 일반론, 한계 고지)은 전부 사전 작성 고정 문구다 (scale_texts.py) - 일반론과
개별 단정의 거리가 한 문장이라 LLM에 맡기지 않고, 고정 문구는 검증도 필요 없다.
결정론 조립 텍스트도 LLM 출력이 아니므로 가드레일과 품질 지표를 거치지 않는다.

길이 원칙: 사람들은 긴 글을 읽지 않는다. 정상 범위 척도는 접힌 한 줄,
준임상·임상 척도만 펼친 카드, 문단은 3문장 상한, 각주·메타는 접기,
첫 화면에서 상담 준비 섹션으로 바로 가는 앵커.
"""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .guardrails import SafeResult, detect_crisis_signals
from .parser import BAND_KO, COMPOSITE_IDS, SYNDROME_IDS, CBCLProfile, SCALE_NAMES
from .renderer import (DEFAULT_SEM, EXAMPLE_RELIABILITY, bell_curve_svg, concept_curve_svg,
                       curve_explainer_svg)
from .scale_texts import GENERAL_OBSERVATION_TEXTS, LIMITS_TEXT, observation_text, scale_card_text

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
# 결정론 조립 블록의 라벨과 태그 (ADR 0010). 태그 문구는 테스트가 개수를 센다.
ASSEMBLED_TAG = "결정론 조립"
OVERVIEW_LABEL = "보호자의 관찰과 검사 소견"
BRIEFING_LABEL = "상담사에게 전달할 요약 미리보기"
# 요약의 질문 절 머리글. 화면의 체크박스와 스크립트로 연동된다 (build_counselor_briefing 독스트링).
BRIEFING_QUESTIONS_NOTE = "위 목록에서 체크한 질문"
LLM_TAG = "LLM 생성 · 검증 통과"


def josa(word: str, with_batchim: str, without: str) -> str:
    """마지막 글자의 받침 유무로 조사를 고른다 (이/가, 은/는)."""
    ch = word[-1]
    if "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28:
        return with_batchim
    return without


def _clean_notes(profile: CBCLProfile) -> list[str]:
    return [n.strip() for n in profile.caregiver_notes if n and n.strip()]


def build_overview_text(profile: CBCLProfile) -> str:
    """연결 문단 (결정론 조립, ADR 0010).

    보호자 의견은 원문 그대로 큰따옴표로 인용하고, 준임상 이상 척도는 보고서 라벨(SCALE_NAMES)
    그대로 밴드별로 나열한 뒤 고정 문장 하나를 붙인다. 전 척도 정상이면 그 사실만 적는다.
    LLM 출력이 아니므로 가드레일·품질 지표의 대상이 아니다.
    """
    notes = _clean_notes(profile)
    if notes:
        first = "보호자님은 이렇게 적어 주셨습니다. " + " ".join(f'"{n}"' for n in notes)
    else:
        first = "보호자 의견은 따로 적히지 않았습니다."
    elevated = profile.elevated_scales()
    if not elevated:
        what = "관찰하신 모습이" if notes else "이 결과가"
        scope = "모든 척도가" if profile.special_scales_administered else "이번 가이드에 포함된 척도는 모두"
        return (f"{first} 검사에서는 {scope} 정상 범위로 보고되었습니다. "
                f"{what} 무엇을 뜻하는지는 상담에서 함께 살펴볼 수 있습니다.")
    parts = []
    for band in ("clinical", "borderline"):
        names = [SCALE_NAMES[s.scale_id] for s in elevated if s.band == band]
        if names:
            parts.append(f"{', '.join(names)}{josa(names[-1], '이', '가')} {BAND_KO[band]} 범위로")
    second = "검사에서는 " + ", ".join(parts) + " 보고되었"
    rest = "그 밖의 척도는" if profile.special_scales_administered else "그 밖의 포함 척도는"
    second += f"고, {rest} 정상 범위였습니다." if len(elevated) < len(profile.all_scales()) else "습니다."
    where = "예약된 상담에서" if profile.counseling_scheduled else "상담 예약 후"
    third = f"이 관찰과 결과가 어떻게 이어지는지는 {where} 상담사와 이야기해 보세요."
    return f"{first} {second} {third}"


OBSERVATION_COUNT = 3


def build_observation_points(profile: CBCLProfile) -> list[dict]:
    """가정 관찰 포인트 3개 (결정론 조립, ADR 0010 보강). 항목은 {"point", "source_scale"}.

    준임상 이상 척도를 위계 순서(종합지표 먼저, 그다음 개별 척도)로 고르되 같은 층 안에서는 임상을
    준임상보다 먼저 두고(임상 척도가 준임상 척도에 밀려 빠지지 않도록), 척도별 고정 문구
    (scale_texts.OBSERVATION_TEXT)를 붙인다. 상승 척도가 셋 미만이면 총 문제행동 기준 일반 문구로
    채우고, 전 척도 정상이면 일반 문구 셋이 전부다. 근거 배지는 그 척도명이다. LLM 출력이
    아니므로 가드레일도 품질 지표도 거치지 않으며, 보호자 의견을 인용하지 않는다 (인용은 질문의 몫).
    """
    order = [*COMPOSITE_IDS, *SYNDROME_IDS]
    ranked = sorted(profile.elevated_scales(),
                    key=lambda s: (s.scale_id not in COMPOSITE_IDS, s.band != "clinical", order.index(s.scale_id)))
    items = [{"point": observation_text(s.scale_id), "source_scale": s.scale_id}
             for s in ranked[:OBSERVATION_COUNT]]
    for text in GENERAL_OBSERVATION_TEXTS:
        if len(items) >= OBSERVATION_COUNT:
            break
        items.append({"point": text, "source_scale": "total_problems"})
    return items


def build_counselor_briefing(profile: CBCLProfile, questions: list, days: int,
                             counseling_scheduled: bool) -> str:
    """상담사에게 전달할 요약 미리보기 (결정론 조립, ADR 0010). 줄바꿈으로 구분된 순수 텍스트.

    보호자 의견 원문, 상승 척도 표(척도, T점수, 보고서 라벨) 또는 "상승 척도 없음", 질문 목록,
    상담까지 남은 일수 또는 미예약 표시. questions는 LLM 출력 항목(dict) 또는 문자열 목록이며,
    비어 있으면 아직 생성되지 않았다고 적는다. LLM 출력이 아니므로 가드레일 대상이 아니다.

    질문 절은 생성 시점에 전체 목록(화면의 체크박스는 기본 전부 체크)이고, 화면에서 체크를 풀면
    템플릿의 스크립트(report.html.j2, data-brief-sync)가 이 텍스트의 같은 번호 줄을 숨기고
    머리글의 개수를 맞춘다. 번호는 화면 목록과 맞추기 위해 다시 매기지 않는다.
    """
    notes = _clean_notes(profile)
    lines: list[str] = []
    lines.append(f"[보호자 의견 원문 {len(notes)}건]" if notes else "[보호자 의견 원문] 적히지 않음")
    lines += [f'{i}. "{n}"' for i, n in enumerate(notes, 1)]
    elevated = profile.elevated_scales()
    if elevated:
        lines.append(f"[상승 척도 {len(elevated)}개] 척도, T점수, 보고서 라벨")
        lines += [f"- {SCALE_NAMES[s.scale_id]} T={s.t_score} {BAND_KO[s.band]}" for s in elevated]
    else:
        lines.append("[상승 척도 없음] 모든 척도 정상 범위")
    texts = [(q.get("question", "") if isinstance(q, dict) else str(q)).strip() for q in questions]
    texts = [t for t in texts if t]
    if texts:
        lines.append(f"[상담사에게 물어볼 질문 {len(texts)}개] {BRIEFING_QUESTIONS_NOTE}")
        lines += [f"{i}. {t}" for i, t in enumerate(texts, 1)]
    else:
        lines.append("[상담사에게 물어볼 질문] 아직 생성되지 않음")
    lines.append(f"[상담까지 {days}일]" if counseling_scheduled else "[상담 미예약]")
    return "\n".join(lines)


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
    """LLM 결과가 아직 없을 때 결정론 부분만 미리 보기 위한 자리표시 SafeResult (prep 1건).

    탐색 콘솔이 슬라이더 변경마다 곡선과 오차 구간 및 고정 문구를 즉시 다시 그리는 데 쓴다.
    생성 블록(질문)은 PENDING_TEXT이고 _pending 표식으로 템플릿이 구분한다. 연결 문단, 관찰 포인트,
    상담사 요약은 결정론 조립이라 미리보기에서도 실제 문구로 나온다.
    """
    return {
        "prep": SafeResult(task="prep", output={
            "questions_for_counselor": [{"question": PENDING_TEXT, "source_scale": None, "_pending": True}]}),
    }


def build_pending_report_html(profile: CBCLProfile, mode_label: str = "생성 전") -> str:
    """생성 전 미리보기: 같은 템플릿·같은 렌더러로 결정론 부분만 실제 값으로 그린다."""
    return build_report_html(profile, pending_results(profile), mode_label=mode_label, pending=True)


def build_preview_html(profile: CBCLProfile, mode_label: str = "생성 전") -> tuple[str, list[str]]:
    """생성 버튼을 누르기 전의 미리보기 화면과 검출된 위기 패턴.

    보호자 의견에 위기 표현이 있으면 입력 게이트(detect_crisis_signals)와 같은 판정으로 점수 리포트
    대신 위기 안내 화면을 돌려준다. 미리보기가 점수와 곡선을 먼저 보여 준 뒤 생성 단계에서야 막는
    구멍을 없애기 위한 것이며, 탐색 콘솔은 입력이 바뀔 때마다 이 함수를 다시 부른다. LLM은 어느
    경로에서도 호출되지 않는다.
    """
    crisis = detect_crisis_signals(profile)
    if crisis:
        return build_crisis_html(profile), crisis
    return build_pending_report_html(profile, mode_label=mode_label), []


def build_report_html(profile: CBCLProfile, results: dict, mode_label: str = "mock",
                      model_label: str = "", pending: bool = False) -> str:
    """SafeResult({"prep"})를 받아 완성 HTML 문자열을 만든다.

    pending=True면 생성 블록 자리에 '생성 대기' 표식을 붙인다 (탐색 콘솔 미리보기).
    연결 문단, 관찰 포인트, 상담사 요약은 여기서 결정론으로 조립한다 (ADR 0010).
    """
    prep = results["prep"]
    fallback_blocks = set(prep.fallback_blocks)
    elevated = [s.name_ko for s in profile.elevated_scales()]
    questions = prep.output["questions_for_counselor"]
    observations = build_observation_points(profile)
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
        overview=build_overview_text(profile),
        overview_label=OVERVIEW_LABEL,
        assembled_tag=ASSEMBLED_TAG,
        llm_tag=LLM_TAG,
        composites=[_scale_view(profile, sid) for sid in COMPOSITE_IDS],
        syndromes=[_scale_view(profile, sid) for sid in SYNDROME_IDS],
        pre_counseling_label=PRE_COUNSELING_LABEL,
        pre_counseling_note=_schedule_aware_text(
            PRE_COUNSELING_NOTE, profile.counseling_scheduled
        ),
        questions=_items_view(profile, questions, "question"),
        observations=_items_view(profile, observations, "point"),
        briefing=build_counselor_briefing(profile, [] if pending else questions,
                                          profile.days_until_counseling, profile.counseling_scheduled),
        briefing_label=BRIEFING_LABEL,
        days=profile.days_until_counseling,
        regen_count=prep.regen_count,
        fallback_count=len(fallback_blocks),
        pending=pending,
    )
