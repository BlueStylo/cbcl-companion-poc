"""동점 비교 화면은 보호자 의견 외 입력이 같을 때만 생성한다."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compare_html import build_compare_html
from src.generator import generate_all
from src.llm_client import MockLLMClient
from src.parser import load_profile, parse_profile


def _profile(name: str):
    return load_profile(ROOT / f"data/profiles/{name}.json")


def _results(profile):
    return generate_all(profile, MockLLMClient())


def test_compare_rejects_different_band_criteria():
    profile_a = _profile("p5a_paired_notes")
    raw = json.loads(
        (ROOT / "data/profiles/p5b_paired_notes.json").read_text(encoding="utf-8")
    )
    raw["band_criteria"]["composite"]["borderline_max_t"] = 63
    profile_b = parse_profile(raw)

    with pytest.raises(ValueError, match="caregiver_notes를 제외한 입력"):
        build_compare_html(profile_a, {}, profile_b, {})


def test_compare_allows_different_notes_profile_id_and_alias():
    profile_a = _profile("p5a_paired_notes")
    profile_b = _profile("p5b_paired_notes").model_copy(
        update={
            "child": _profile("p5b_paired_notes").child.model_copy(
                update={"alias": "다른가명"}
            )
        }
    )

    html = build_compare_html(
        profile_a,
        _results(profile_a),
        profile_b,
        _results(profile_b),
    )

    assert profile_a.caregiver_notes[0] in html
    assert profile_b.caregiver_notes[0] in html
