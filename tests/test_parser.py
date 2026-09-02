"""파서 결정론 테스트 (LLM 호출 없음)."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser import KCBCL_DEFAULT_BAND_CRITERIA, ProfileError, load_profile, parse_profile

PROFILES = sorted((ROOT / "data" / "profiles").glob("*.json"))


@pytest.fixture()
def base_raw():
    return json.loads((ROOT / "data/profiles/p1_all_normal.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", PROFILES, ids=[p.stem for p in PROFILES])
def test_valid_profiles_pass(path):
    profile = load_profile(path)
    assert len(profile.composites) == 3
    assert len(profile.syndromes) == 8


def test_t_score_out_of_range_rejected(base_raw):
    raw = copy.deepcopy(base_raw)
    raw["syndromes"][0]["t_score"] = 120
    with pytest.raises(ProfileError):
        parse_profile(raw)


def test_band_criteria_below_supported_range_rejected(base_raw):
    raw = copy.deepcopy(base_raw)
    raw["band_criteria"]["composite"]["normal_max_t"] = 49
    with pytest.raises(ProfileError, match="greater than or equal to 50"):
        parse_profile(raw)


def test_band_criteria_above_supported_range_rejected_and_defaults_documented(base_raw):
    raw = copy.deepcopy(base_raw)
    raw["band_criteria"]["syndrome"]["borderline_max_t"] = 91
    with pytest.raises(ProfileError, match="less than or equal to 90"):
        parse_profile(raw)
    assert KCBCL_DEFAULT_BAND_CRITERIA == {
        "composite": {"normal_max_t": 59, "borderline_max_t": 62},
        "syndrome": {"normal_max_t": 59, "borderline_max_t": 69},
    }


def test_band_label_mismatch_rejected(base_raw):
    """T=55인데 준임상 라벨 - 밴드 재계산 대조가 잡아야 한다."""
    raw = copy.deepcopy(base_raw)
    raw["syndromes"][1]["band"] = "borderline"
    with pytest.raises(ProfileError, match="borderline"):
        parse_profile(raw)


def test_missing_syndrome_rejected(base_raw):
    raw = copy.deepcopy(base_raw)
    raw["syndromes"] = raw["syndromes"][:7]
    with pytest.raises(ProfileError):
        parse_profile(raw)


def test_unknown_scale_id_rejected(base_raw):
    raw = copy.deepcopy(base_raw)
    raw["composites"][0]["scale_id"] = "unknown_scale"
    with pytest.raises(ProfileError):
        parse_profile(raw)


def test_wrong_scale_name_rejected(base_raw):
    """공식 척도명과 다른 name_ko는 거부된다."""
    raw = copy.deepcopy(base_raw)
    raw["syndromes"][0]["name_ko"] = "다른이름"
    with pytest.raises(ProfileError):
        parse_profile(raw)


def test_llm_payload_masks_alias_in_notes():
    """LLM 입력에는 아동 이름이 필드로도, 보호자 의견 본문으로도 들어가지 않는다."""
    from src.generator import mask_notes, profile_payload
    profile = load_profile(ROOT / "data/profiles/p2_partial_borderline.json")
    alias = profile.child.alias
    notes = [f"{alias}가 요즘 숙제를 미룹니다", "놀이터에서 또래를 피합니다"]
    assert mask_notes(notes, alias) == ["아이가 요즘 숙제를 미룹니다", "놀이터에서 또래를 피합니다"]
    payload = profile_payload(profile)
    assert alias not in json.dumps(payload, ensure_ascii=False)
    assert "alias" not in payload["child"]
