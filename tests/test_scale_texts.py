"""척도 x 밴드 고정 문구 테스트: 완전성, 길이 상한, 가드레일 어휘 통과."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.guardrails import _check_text
from src.parser import BAND_KO, COMPOSITE_IDS, SYNDROME_IDS, load_profile
from src.scale_texts import (BANDS, LIMITS_TEXT, SCALE_BAND_TEXT, scale_card_text,
                             scale_one_liner)

PROFILE = load_profile(ROOT / "data/profiles/p2_partial_borderline.json")


def test_every_scale_band_has_text():
    assert len(SCALE_BAND_TEXT) == 11 * 3
    for sid in (*COMPOSITE_IDS, *SYNDROME_IDS):
        for band in BANDS:
            assert scale_card_text(sid, band).strip()


def test_fixed_texts_are_at_most_three_sentences():
    for text in (*SCALE_BAND_TEXT.values(), LIMITS_TEXT):
        assert text.count(". ") + 1 <= 3, text


def test_fixed_texts_pass_guardrail_vocabulary():
    """고정 문구도 LLM 출력과 같은 어휘 규칙(G1/G2/G6/G7/G8)을 만족한다.

    G3(수치)는 대상이 아니고, G8 라벨 대조는 문구가 자기 밴드 라벨만 쓰는지로 본다.
    """
    for (sid, band), text in SCALE_BAND_TEXT.items():
        vs = [v for v in _check_text(f"{sid}:{band}", text, PROFILE) if v.rule_id != "G3"]
        vs = [v for v in vs if not (v.rule_id == "G8" and "라벨 불일치" in v.matched)]
        assert vs == [], (sid, band, [v.matched for v in vs])
        assert BAND_KO[band] in text
        others = {"정상", "준임상", "임상"} - {BAND_KO[band]}
        for other in others:
            if other == "임상":
                assert "준임상" in text or "임상" not in text.replace("준임상", "")
            else:
                assert other not in text.replace("준임상", "") if other == "정상" else other not in text


def test_one_liner_mentions_name_and_label():
    assert scale_one_liner("withdrawn", "normal").startswith("위축 · 정상")
