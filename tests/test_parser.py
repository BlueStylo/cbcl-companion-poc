"""파서 결정론 테스트 (LLM 호출 없음)."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser import ProfileError, load_profile, parse_profile

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
