"""입력 CBCL 프로파일 JSON의 검증과 구조화 (결정론 1차 관문).

형식 오류, 값 범위 오류, 라벨-수치 불일치 중 하나라도 있으면 ProfileError를
던지고 리포트 생성을 시작하지 않는다 (fail-closed). 밴드 판정의 원천은 검사
기관의 기준표이고, 이 모듈은 그 기준을 재계산해 대조만 한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

Band = Literal["normal", "borderline", "clinical"]

BAND_KO = {"normal": "정상", "borderline": "준임상", "clinical": "임상"}

# 리포트 2페이지의 위계 순서: 총점부터 읽는다
COMPOSITE_IDS = ("total_problems", "internalizing", "externalizing")
SYNDROME_IDS = (
    "withdrawn", "somatic", "anxious_depressed", "social_immaturity",
    "thought_problems", "attention", "delinquent", "aggressive",
)

# K-CBCL 표준 척도명 (검사 도구의 공식 용어)
SCALE_NAMES = {
    "total_problems": "총 문제행동",
    "internalizing": "내재화 문제",
    "externalizing": "외현화 문제",
    "withdrawn": "위축",
    "somatic": "신체증상",
    "anxious_depressed": "우울/불안",
    "social_immaturity": "사회적 미성숙",
    "thought_problems": "사고의 문제",
    "attention": "주의집중",
    "delinquent": "비행",
    "aggressive": "공격성",
}

# 척도의 교과서적 정의 (고정 문구 - 화면과 폴백 안전 문구에서 공용)
SCALE_DEFINITIONS = {
    "total_problems": "문제행동 문항 전체를 합산한 종합 점수로, 결과를 읽는 출발점입니다.",
    "internalizing": "위축, 신체증상, 우울/불안처럼 안으로 향하는 어려움을 묶은 종합지표입니다.",
    "externalizing": "비행, 공격성처럼 밖으로 드러나는 행동을 묶은 종합지표입니다.",
    "withdrawn": "혼자 있으려 하거나 또래와의 접촉을 피하는 모습이 얼마나 자주 보고되었는지를 봅니다.",
    "somatic": "뚜렷한 의학적 원인 없이 보고되는 몸의 불편감(두통, 복통 등)을 봅니다.",
    "anxious_depressed": "걱정이 많아 보이거나 기분이 가라앉아 보이는 모습이 얼마나 보고되었는지를 봅니다.",
    "social_immaturity": "나이에 비해 어리게 행동하거나 또래 관계에서 서툰 모습을 봅니다.",
    "thought_problems": "반복 행동이나 주변이 이해하기 어려운 생각 표현이 보고되었는지를 봅니다.",
    "attention": "집중을 유지하기 어렵거나 가만히 있기 어려운 모습이 얼마나 보고되었는지를 봅니다.",
    "delinquent": "규칙을 어기는 행동이 얼마나 보고되었는지를 봅니다.",
    "aggressive": "말다툼, 떼쓰기, 공격적인 행동이 얼마나 보고되었는지를 봅니다.",
}

T_MIN, T_MAX = 30, 100


class ProfileError(ValueError):
    """입력 프로파일 검증 실패. errors에 사유 목록을 담는다."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class ScaleScore(BaseModel):
    scale_id: str
    name_ko: str
    t_score: int = Field(ge=T_MIN, le=T_MAX)  # T점수 물리적 범위
    band: Band


class BandCriteria(BaseModel):
    normal_max_t: int
    borderline_max_t: int

    @model_validator(mode="after")
    def check_order(self):
        if self.normal_max_t >= self.borderline_max_t:
            raise ValueError("band_criteria: normal_max_t < borderline_max_t 이어야 함")
        return self


class ChildInfo(BaseModel):
    alias: str  # 명백한 가상명만 사용. LLM 입력에는 넘기지 않는다.
    sex: Literal["male", "female"]
    age_years: int = Field(ge=4, le=18)
    age_months: int = Field(ge=0, le=11)
    norm_group: str


def expected_band(t_score: int, criteria: BandCriteria) -> str:
    """기준표로 밴드를 재계산한다. 판정의 유일한 계산 지점."""
    if t_score <= criteria.normal_max_t:
        return "normal"
    if t_score <= criteria.borderline_max_t:
        return "borderline"
    return "clinical"


class CBCLProfile(BaseModel):
    profile_id: str
    profile_type: Literal[
        "all_normal", "partial_borderline", "boundary_mix",
        "clinical", "paired_notes", "adversarial",
    ]
    instrument: str
    child: ChildInfo
    test_date: str
    band_criteria: dict[Literal["composite", "syndrome"], BandCriteria]
    composites: list[ScaleScore] = Field(min_length=3, max_length=3)
    syndromes: list[ScaleScore] = Field(min_length=8, max_length=8)
    special_scales_administered: bool = False
    caregiver_notes: list[str] = Field(default_factory=list, max_length=5)
    counseling_scheduled: bool = True
    days_until_counseling: int = Field(ge=0, le=60)

    @model_validator(mode="after")
    def check_scale_sets(self):
        """척도 id 집합과 공식 척도명이 표준과 일치하는지 검사."""
        for kind, scales, want in (
            ("composites", self.composites, COMPOSITE_IDS),
            ("syndromes", self.syndromes, SYNDROME_IDS),
        ):
            got = {s.scale_id for s in scales}
            if got != set(want):
                raise ValueError(f"{kind}: 척도 id 집합이 표준과 다름 (누락: {set(want) - got}, 초과: {got - set(want)})")
            for s in scales:
                if s.name_ko != SCALE_NAMES[s.scale_id]:
                    raise ValueError(f"{s.scale_id}: name_ko={s.name_ko!r}는 표준 척도명 {SCALE_NAMES[s.scale_id]!r}과 다름")
        return self

    @model_validator(mode="after")
    def check_band_consistency(self):
        """선언된 밴드 라벨을 기준표로 재계산해 대조. 이 모듈의 핵심."""
        for kind, scales in (("composite", self.composites), ("syndrome", self.syndromes)):
            criteria = self.band_criteria.get(kind)
            if criteria is None:
                raise ValueError(f"band_criteria에 {kind} 기준이 없음")
            for s in scales:
                want = expected_band(s.t_score, criteria)
                if s.band != want:
                    raise ValueError(
                        f"{s.scale_id}: T={s.t_score}인데 band={s.band}. 기준상 {want}이어야 함")
        return self

    # ---- 이후 단계(렌더러, 가드레일)가 쓰는 헬퍼 ----

    def all_scales(self) -> list[ScaleScore]:
        return list(self.composites) + list(self.syndromes)

    def scale_map(self) -> dict[str, ScaleScore]:
        return {s.scale_id: s for s in self.all_scales()}

    def kind_of(self, scale_id: str) -> str:
        return "composite" if scale_id in COMPOSITE_IDS else "syndrome"

    def criteria_for(self, scale_id: str) -> BandCriteria:
        return self.band_criteria[self.kind_of(scale_id)]

    def elevated_scales(self) -> list[ScaleScore]:
        """준임상/임상 범위 척도 (위계 순서)."""
        order = list(COMPOSITE_IDS) + list(SYNDROME_IDS)
        m = self.scale_map()
        return [m[sid] for sid in order if m[sid].band != "normal"]

    def allowed_numbers(self) -> set[int]:
        """LLM 출력 본문에 등장해도 되는 수치 집합 (가드레일 G3의 기준)."""
        allowed = {s.t_score for s in self.all_scales()}
        for c in self.band_criteria.values():
            allowed |= {c.normal_max_t, c.normal_max_t + 1,
                        c.borderline_max_t, c.borderline_max_t + 1}
        # T점수 체계 설명용 상수 (평균 50, 표준편차 10, 통상 범위 40~60, SEM 68%)
        allowed |= {10, 30, 40, 50, 60, 68, 70, 80, 100}
        return allowed


def parse_profile(raw: dict) -> CBCLProfile:
    """dict를 검증해 CBCLProfile로. 실패 시 ProfileError (fail-closed)."""
    try:
        return CBCLProfile.model_validate(raw)
    except ValidationError as e:
        msgs = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
        raise ProfileError(msgs) from e


def load_profile(path: str | Path) -> CBCLProfile:
    """프로파일 JSON 파일을 읽어 검증한다."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ProfileError([f"{p.name}: JSON을 읽을 수 없음 ({e})"]) from e
    return parse_profile(raw)
