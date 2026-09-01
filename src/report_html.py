"""2페이지 정적 HTML 리포트 생성.

1페이지 '관찰자의 렌즈'(고정 문구 + 개념 예시 곡선), 2페이지 '우리 아이
결과'(총점 위계 + 척도 카드), 이어서 상담 준비 도우미 섹션. 서버 없이
브라우저로 열면 끝나는 단일 파일이며 외부 CDN을 쓰지 않는다.

고정 심리교육 문구(렌즈 안내, T점수 설명, 준임상 일반론, 유의사항)는
사전 작성 문구를 그대로 노출한다. 일반론과 개별 단정의 거리가 한 문장이라
이 문단들은 LLM에 맡기지 않는다.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .parser import BAND_KO, COMPOSITE_IDS, SYNDROME_IDS, CBCLProfile, SCALE_NAMES
from .renderer import DEFAULT_SEM, EXAMPLE_RELIABILITY, bell_curve_svg, concept_curve_svg

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# ---- 고정 문구 (사전 작성, LLM 미사용) ----
LENS_QUOTE = (
    "이 결과는 보호자의 눈으로 본 아이입니다. 관찰자가 다르면(선생님, 아이 자신) "
    "결과가 다르게 나오는 것이 정상이고, 그 차이 자체가 중요한 정보입니다. "
    "그래서 상담사와의 대화에서 그림이 완성됩니다."
)
TSCORE_EXPLAIN = (
    "T점수는 또래 집단의 평균을 50, 표준편차를 10으로 맞춘 점수입니다. "
    "또래 대부분은 40에서 60 사이에 위치합니다. 점수는 점이 아니라 구간으로 "
    "읽는 것이 정확하며, 아래 곡선의 밴드가 그 구간입니다."
)
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
    f"마커 주변의 밴드는 측정의 표준오차(SEM)로 계산한 대칭 구간입니다. "
    f"신뢰도 예시값 {EXAMPLE_RELIABILITY} 기준 SEM {DEFAULT_SEM}T이며(예시값 - 실제 서비스는 "
    f"검사 매뉴얼의 척도별 신뢰도 계수 사용), 반복 측정 시 점수가 이 범위에 들어올 확률은 약 68%입니다."
)
BORDERLINE_NOTE = (
    "준임상 구간은 확정된 상태가 아니라 관찰과 개입의 여지가 있는 구간입니다. "
    "이 검사는 조기 발견을 위한 선별 도구이며, 해석 유의사항은 재검사와 다중 정보원"
    "(교사 보고 TRF, 자기 보고 YSR) 병행을 권고합니다."
)
CAUTION = "이 보고서는 선별 도구이며 진단이 아닙니다. 단일 검사 결과만으로 진단을 확정할 수 없습니다."


def _template_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(enabled_extensions=("html", "j2")),
    )


def _scale_view(profile: CBCLProfile, sid: str, exp_map: dict) -> dict:
    """템플릿에 넘길 척도 카드 1개 분량의 데이터."""
    scale = profile.scale_map()[sid]
    item = exp_map.get(sid, {})
    return {
        "name": scale.name_ko,
        "t": scale.t_score,
        "band": scale.band,
        "band_ko": BAND_KO[scale.band],
        "svg": bell_curve_svg(scale.t_score, profile.criteria_for(sid)),
        "what_it_measures": item.get("what_it_measures", ""),
        "what_the_number_means": item.get("what_the_number_means", ""),
        "everyday_example": item.get("everyday_example", ""),
        "fallback": bool(item.get("_fallback")),
    }


def _items_view(profile: CBCLProfile, items: list, text_key: str) -> list[dict]:
    names = SCALE_NAMES
    return [{"text": it.get(text_key, ""),
             "source_name": names.get(it.get("source_scale"), "원 보고서")}
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


def build_report_html(profile: CBCLProfile, results: dict, mode_label: str = "mock") -> str:
    """SafeResult 2건({"explain","prep"})을 받아 완성 HTML 문자열을 만든다."""
    explain, prep = results["explain"], results["prep"]
    exp_map = {i.get("scale_id"): i for i in explain.output["scale_explanations"]
               if isinstance(i, dict)}

    fallback_blocks = set(explain.fallback_blocks) | set(prep.fallback_blocks)
    return _template_env().get_template("report.html.j2").render(
        alias=profile.child.alias,
        instrument=profile.instrument,
        test_date=profile.test_date,
        mode_label=mode_label,
        lens_quote=LENS_QUOTE,
        tscore_explain=TSCORE_EXPLAIN,
        footnotes=FOOTNOTES,
        concept_svg=concept_curve_svg(),
        sem_note=SEM_NOTE,
        borderline_note=BORDERLINE_NOTE,
        caution=CAUTION,
        overview=explain.output["overview"],
        overview_fallback="overview" in fallback_blocks,
        composites=[_scale_view(profile, sid, exp_map) for sid in COMPOSITE_IDS],
        syndromes=[_scale_view(profile, sid, exp_map) for sid in SYNDROME_IDS],
        limits=explain.output["limits"],
        limits_fallback="limits" in fallback_blocks,
        before_counseling=explain.output["before_counseling"],
        questions=_items_view(profile, prep.output["questions_for_counselor"], "question"),
        observations=_items_view(profile, prep.output["observation_points"], "point"),
        briefing=prep.output["counselor_briefing"],
        days=profile.days_until_counseling,
        regen_count=explain.regen_count + prep.regen_count,
        fallback_count=len(fallback_blocks),
    )
