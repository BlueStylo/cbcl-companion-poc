"""품질 지표 3종 테스트 (LLM 호출 없음, 측정 도구 자체 검증)."""

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser import load_profile
from src.quality import (direction_warnings, jargon_metrics, note_tokens,
                         quality_summary, reflection_metrics)

PROFILE = load_profile(ROOT / "data/profiles/p2_partial_borderline.json")
FIXTURE = json.loads((ROOT / "data/fixtures/p2_partial_borderline.json").read_text(encoding="utf-8"))


def test_note_tokens_strip_particles_and_keep_stems():
    tokens = note_tokens(["학원 숙제를 앞에 두면 딴 데를 자주 봅니다",
                          "놀이터에서 또래에게 먼저 말을 거는 일이 줄었습니다"])
    for want in ("학원", "숙제", "놀이터", "또래", "먼저"):
        assert want in tokens
    assert "자주" not in tokens          # 불용어
    assert all(len(t) >= 2 for t in tokens)


def test_reflection_rate_on_clean_fixture_is_high():
    """mock 픽스처는 보호자 문장을 그대로 인용하므로 반영률이 높아야 한다."""
    r = reflection_metrics(PROFILE, FIXTURE["prep"]["attempts"][0])
    assert r["items_total"] == 10
    assert r["item_rate"] >= 0.5
    assert {"숙제", "또래"} <= set(r["tokens_hit"])


def test_reflection_rate_drops_for_generic_questions():
    out = copy.deepcopy(FIXTURE["prep"]["attempts"][0])
    for it in out["questions_for_counselor"]:
        it["question"] = "이 결과를 어떻게 보면 될까요?"
    for it in out["observation_points"]:
        it["point"] = "하루 한 줄 메모하기"
    assert reflection_metrics(PROFILE, out)["items_reflected"] == 0


def test_jargon_metrics_count_terms_and_gloss():
    texts = [("a", "T점수 57는 정상 범위입니다. 또래 100명 중 어디쯤인지로 읽으면 됩니다."),
             ("b", "내재화 문제와 외현화 문제는 척도 묶음입니다."),
             ("c", "아이의 하루를 한 줄로 적어 두세요.")]
    j = jargon_metrics(texts)
    assert j["blocks_total"] == 3 and j["blocks_with_term"] == 2
    assert j["glossed_blocks"] == 1
    assert j["by_term"]["T점수"] == 1 and j["by_term"]["척도"] == 1
    assert "준임상" not in j["by_term"] and j["by_term"].get("임상") is None


def test_direction_warnings_flag_reverse_questions_only():
    """실LLM 실측 결함 (a): 상담사가 보호자에게 되묻는 형태를 WARN으로 집계."""
    out = copy.deepcopy(FIXTURE["prep"]["attempts"][0])
    assert direction_warnings(out) == []            # 픽스처(보호자→상담사)는 경고 0
    out["questions_for_counselor"][0]["question"] = "위축 증상이 관찰되는 상황이나 행동 사례를 몇 가지 더 알려주시겠습니까?"
    out["questions_for_counselor"][1]["question"] = "신체 증상이 언제 나타나는지 구체적으로 알려주시면 도움이 될 것 같습니다."
    out["questions_for_counselor"][2]["question"] = "위축된 모습을 관찰하신 구체적인 사례가 있으신가요?"
    warns = direction_warnings(out)
    assert [w["block"] for w in warns] == ["questions_for_counselor[0]", "questions_for_counselor[1]",
                                           "questions_for_counselor[2]"]
    # 보호자→상담사 요청문("알려 주세요")은 방향이 같으므로 경고하지 않는다 - 의미 판정은 규칙 한계
    out = copy.deepcopy(FIXTURE["prep"]["attempts"][0])
    out["questions_for_counselor"][0]["question"] = "두 결과를 함께 읽는 방법을 알려 주세요."
    assert direction_warnings(out) == []


def test_quality_summary_excludes_fallback_items():
    out = copy.deepcopy(FIXTURE["prep"]["attempts"][0])
    out["questions_for_counselor"] = [{"question": "폴백 (scale_id: attention) 알려주시겠어요?",
                                       "source_scale": "total_problems", "_fallback": True}]
    q = quality_summary(PROFILE, {"explain": FIXTURE["explain"]["attempts"][0], "prep": out})
    assert q["direction_warnings"] == []
    assert q["reflection"]["items_total"] == 4       # 관찰 포인트만 남는다
