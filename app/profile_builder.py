"""탐색 콘솔의 입력 조립 (Streamlit 비의존, 테스트 대상).

슬라이더·텍스트 값 → 프로파일 dict → 기존 파서(parse_profile)를 그대로 통과시킨다.
밴드 라벨은 여기서 따로 계산하지 않고 파서의 expected_band(개별 60/70, 종합 60/63)로
채우므로 규칙은 한 벌이다. 종합 지표는 하위 척도에서 유도하지 않고 직접 입력받는다
(COMPOSITE_NOTE 참조).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from src.parser import (COMPOSITE_IDS, KCBCL_DEFAULT_BAND_CRITERIA, SCALE_NAMES,
                        SYNDROME_IDS, BandCriteria, CBCLProfile, expected_band,
                        parse_profile)

ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = ROOT / "data" / "profiles"

T_SLIDER_MIN, T_SLIDER_MAX = 30, 90
ALL_SCALE_IDS = (*COMPOSITE_IDS, *SYNDROME_IDS)

# 기준표 (data/profiles 8종과 동일). 실서비스는 검사 시스템의 기준표를 받는다.
DEFAULT_CRITERIA = json.loads(json.dumps(KCBCL_DEFAULT_BAND_CRITERIA))

# 종합 지표를 구성하는 하위 척도 (참고 힌트 계산에만 쓴다 - 값을 유도하지 않는다)
COMPOSITE_MEMBERS = {
    "internalizing": ("withdrawn", "somatic", "anxious_depressed"),
    "externalizing": ("delinquent", "aggressive"),
    "total_problems": SYNDROME_IDS,
}

COMPOSITE_NOTE = (
    "종합 지표는 하위 척도 평균이 아니라 문항 원점수의 규준 변환값입니다. "
    "규준 변환은 검사 시스템의 몫이므로 이 콘솔은 결과 보고서의 수치를 그대로 입력받습니다."
)

EXAMPLE_ORDER = [
    "p1_all_normal", "p2_partial_borderline", "p3_boundary_mix", "p4_clinical",
    "p5a_paired_notes", "p5b_paired_notes", "a1_adversarial", "c1_crisis",
]
EXAMPLE_LABELS = {
    "p1_all_normal": "P1 · 전 척도 정상",
    "p2_partial_borderline": "P2 · 부분 준임상 (내재화·위축·우울/불안·주의집중)",
    "p3_boundary_mix": "P3 · 기준선 인접 혼합 (총점 60 · 공격성 70)",
    "p4_clinical": "P4 · 임상 (총점·외현화·공격성)",
    "p5a_paired_notes": "P5a · 동점 페어 A (위축·속상함 의견)",
    "p5b_paired_notes": "P5b · 동점 페어 B (준비물·집중 의견)",
    "a1_adversarial": "A1 · 위반 유도 의견 (진단명·판정 요구)",
    "c1_crisis": "C1 · 위기 신호 의견 (LLM 미호출 게이트)",
}


@dataclass
class ExplorerInputs:
    """화면 위젯 값 한 벌. t_scores는 척도 11개 전부."""
    t_scores: dict[str, int]
    notes: list[str] = field(default_factory=list)
    alias: str = "김샘플"
    sex: str = "female"
    age_years: int = 9
    age_months: int = 0
    days_until_counseling: int = 5
    test_date: str = ""
    criteria: dict = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_CRITERIA)))


def norm_group(sex: str, age_years: int) -> str:
    return f"{'남자' if sex == 'male' else '여자'} {'4-11세' if age_years <= 11 else '12-18세'}"


def kind_of(scale_id: str) -> str:
    return "composite" if scale_id in COMPOSITE_IDS else "syndrome"


def band_for(scale_id: str, t_score: int, criteria: dict = DEFAULT_CRITERIA) -> str:
    """밴드 라벨은 파서 규칙(expected_band) 그대로."""
    return expected_band(t_score, BandCriteria(**criteria[kind_of(scale_id)]))


def profile_type_for(t_scores: dict[str, int], criteria: dict) -> str:
    bands = {band_for(sid, t, criteria) for sid, t in t_scores.items()}
    if bands == {"normal"}:
        return "all_normal"
    return "clinical" if "clinical" in bands else "partial_borderline"


def build_profile_raw(inputs: ExplorerInputs) -> dict:
    """위젯 값 → 파서 입력 dict. 검증은 하지 않는다 (파서가 한다)."""
    crit = inputs.criteria

    def entry(sid: str) -> dict:
        t = int(inputs.t_scores[sid])
        return {"scale_id": sid, "name_ko": SCALE_NAMES[sid], "t_score": t, "band": band_for(sid, t, crit)}

    return {
        "profile_id": "explorer",
        "profile_type": profile_type_for(inputs.t_scores, crit),
        "instrument": "K-CBCL 6-18 (보호자 보고형)",
        "child": {
            "alias": (inputs.alias or "김샘플").strip(),
            "sex": inputs.sex,
            "age_years": int(inputs.age_years),
            "age_months": int(inputs.age_months),
            "norm_group": norm_group(inputs.sex, int(inputs.age_years)),
        },
        "test_date": inputs.test_date or date.today().isoformat(),
        "band_criteria": crit,
        "composites": [entry(sid) for sid in COMPOSITE_IDS],
        "syndromes": [entry(sid) for sid in SYNDROME_IDS],
        "special_scales_administered": False,
        "caregiver_notes": [n.strip() for n in inputs.notes if n and n.strip()][:5],
        "counseling_scheduled": True,
        "days_until_counseling": int(inputs.days_until_counseling),
    }


def build_profile(inputs: ExplorerInputs) -> CBCLProfile:
    """위젯 값 → 검증된 프로파일. 실패 시 ProfileError (fail-closed, 파서와 동일)."""
    return parse_profile(build_profile_raw(inputs))


def example_inputs(profile_id: str) -> ExplorerInputs:
    """data/profiles의 예시 1건을 위젯 값으로 푼다 (슬라이더에 로드)."""
    raw = json.loads((PROFILES_DIR / f"{profile_id}.json").read_text(encoding="utf-8"))
    return ExplorerInputs(
        t_scores={s["scale_id"]: s["t_score"] for s in raw["composites"] + raw["syndromes"]},
        notes=list(raw["caregiver_notes"]),
        alias=raw["child"]["alias"],
        sex=raw["child"]["sex"],
        age_years=raw["child"]["age_years"],
        age_months=raw["child"]["age_months"],
        days_until_counseling=raw["days_until_counseling"],
        test_date=raw["test_date"],
        criteria=raw["band_criteria"],
    )


def composite_hints(t_scores: dict[str, int], criteria: dict = DEFAULT_CRITERIA) -> list[str]:
    """부드러운 참고 힌트 (차단 아님): 하위 척도가 전부 준임상 이상인데 대응 종합 지표가
    정상이면 한 줄. 종합 지표는 원점수 규준 변환값이라 이런 조합도 가능하므로 확인만 권한다."""
    hints = []
    for cid, members in COMPOSITE_MEMBERS.items():
        if band_for(cid, t_scores[cid], criteria) != "normal":
            continue
        if all(band_for(m, t_scores[m], criteria) != "normal" for m in members):
            what = "개별 척도 8개" if cid == "total_problems" \
                else "하위 척도(" + ", ".join(SCALE_NAMES[m] for m in members) + ")"
            hints.append(f"{what}는 모두 준임상 이상인데 종합 지표 '{SCALE_NAMES[cid]}'은(는) 정상 범위로 입력되어 "
                         "있습니다. 규준 변환값이라 가능한 조합이지만, 결과 보고서의 수치가 맞는지 한 번 확인해 보세요.")
    return hints
