"""동점-상이의견 페어(P5a/P5b) 나란히 비교 HTML.

동일한 T점수 프로파일에서 보호자 의견 텍스트만 다를 때 질문과 관찰 포인트가
실제로 달라지는지를 한 화면에서 보여준다. "LLM이 템플릿으로 대체되지
않는 지점"의 실증이 곧 데모다. 연결 문단은 결정론 조립(ADR 0010)이라 의견 인용만 다르다.
"""

from __future__ import annotations

from .parser import BAND_KO, COMPOSITE_IDS, SYNDROME_IDS, CBCLProfile
from .report_html import _items_view, _template_env, build_overview_text


def _comparison_input(profile: CBCLProfile) -> dict:
    """비교에서 달라도 되는 식별자와 보호자 의견만 제거한 입력을 만든다."""
    data = profile.model_dump()
    data.pop("profile_id")
    data.pop("caregiver_notes")
    data["child"].pop("alias")
    return data


def build_compare_html(profile_a: CBCLProfile, results_a: dict,
                       profile_b: CBCLProfile, results_b: dict,
                       mode_label: str = "mock") -> str:
    """두 프로파일과 생성 결과를 받아 비교 HTML 문자열을 만든다.

    식별자와 보호자 의견을 제외한 생성 입력이 같은지 먼저 확인한다.
    """
    if _comparison_input(profile_a) != _comparison_input(profile_b):
        raise ValueError(
            "비교 대상은 profile_id, alias, caregiver_notes를 제외한 입력이 동일해야 합니다"
        )

    m = profile_a.scale_map()
    shared_scales = [{"name": m[sid].name_ko, "t": m[sid].t_score,
                      "band": m[sid].band, "band_ko": BAND_KO[m[sid].band]}
                     for sid in (*COMPOSITE_IDS, *SYNDROME_IDS)]

    def case_view(profile: CBCLProfile, results: dict) -> dict:
        return {
            "profile_id": profile.profile_id,
            "notes": list(profile.caregiver_notes),
            "overview": build_overview_text(profile),   # 결정론 조립 (ADR 0010)
            "questions": _items_view(profile, results["prep"].output["questions_for_counselor"], "question"),
            "observations": _items_view(profile, results["prep"].output["observation_points"], "point"),
        }

    return _template_env().get_template("compare.html.j2").render(
        mode_label=mode_label,
        shared_scales=shared_scales,
        cases=[case_view(profile_a, results_a), case_view(profile_b, results_b)],
    )
