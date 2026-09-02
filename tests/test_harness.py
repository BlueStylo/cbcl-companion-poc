"""하네스 자체의 결정론 테스트 (LLM 호출 없음).

하네스가 지표를 재는 도구인 만큼, 도구 자신도 검증한다: 뮤테이션 적용,
미검출(MISS) 집계, 원시/최종 커버리지 구분, 시드 재고 대조.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

import run_harness as rh
from src.generator import generate_all
from src.guardrails import source_coverage
from src.llm_client import MockLLMClient
from src.parser import load_profile


def test_apply_mutations_four_ops():
    """set / delete / append / truncate와 중첩 경로 해석."""
    out = {
        "overview": "이전",
        "questions": [{"question": "q", "x": 1}],
        "items": [1, 2, 3],
    }
    rh.apply_mutations(out, [
        {"op": "set", "path": ["overview"], "value": "이후"},
        {"op": "set", "path": ["questions", 0, "y"], "value": 9},
        {"op": "delete", "path": ["questions", 0, "x"]},
        {"op": "append", "path": ["items"], "value": 4},
        {"op": "truncate", "path": ["items"], "value": 2},
    ])
    assert out["overview"] == "이후"
    assert out["questions"][0] == {"question": "q", "y": 9}
    assert out["items"] == [1, 2]


def test_undetectable_seed_counted_as_miss(tmp_path, monkeypatch):
    """검출될 수 없는 기대 규칙(뮤테이션 없음 + G1 기대)은 MISS로 집계."""
    case_file = {"category": "테스트", "cases": [{
        "id": "impossible", "task": "prep",
        "profile_id": "p1_all_normal", "note": "",
        "expect_rules": ["G1"], "mutations": [],
    }]}
    (tmp_path / "t.json").write_text(
        json.dumps(case_file, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(rh, "SEEDED_DIR", tmp_path)
    monkeypatch.setattr(rh, "SEEDED_FILE_ORDER", ["t"])

    hit, total, rows, fails = rh.run_seeded_check()
    assert (hit, total) == (0, 1)
    assert rows[0][-1] == "MISS"
    assert fails and "impossible" in fails[0]


def test_raw_vs_final_coverage_are_different_metrics():
    """원시 attempt 0 커버리지는 생성 품질(<100% 가능), 최종 출력은

    fail-closed 조립이라 구조상 항상 100%임을 A1로 확인한다. 근거 필드는
    prep에만 있으므로 다른 task 이름은 0/0이다.
    """
    assert source_coverage(load_profile(ROOT / "data/profiles/p2_partial_borderline.json"),
                           "explain", {"overview": "x"}) == (0, 0)
    profile = load_profile(ROOT / "data/profiles/a1_adversarial.json")
    client = MockLLMClient()

    raw = rh._raw_attempt0(client, profile, "prep")
    raw_have, raw_need = source_coverage(profile, "prep", raw)
    assert raw_have < raw_need  # 원시 출력에는 근거 위반 시드가 있다

    final = generate_all(profile, client)["prep"].output
    final_have, final_need = source_coverage(profile, "prep", final)
    assert final_have == final_need


def test_seed_inventory_matches_gate_constants():
    """시드 파일 재고: B축 45건(G1~G11 + 우회), 파이프라인 10건, expect_rules 공백 없음, task는 prep뿐.

    새 규칙(G3 숫자·한글 수사, G5 문형, G10 근거 없음·척도 불일치, G11 방향)에는 각각 2건 이상이 있어야 한다.
    검토 반영분(G10 인용 주장 변형 2, G3 점수 어휘 뒤 수사 1, G6 조사 변형 1)으로 41건에서 45건이 됐다.
    """
    total = 0
    by_file = {}
    notes = []
    for name in rh.SEEDED_FILE_ORDER:
        data = json.loads(
            (rh.SEEDED_DIR / f"{name}.json").read_text(encoding="utf-8"))
        for case in data["cases"]:
            assert case["expect_rules"], f"{case['id']}: expect_rules 비어 있음"
            assert case["task"] == "prep", case["id"]
            notes.append(case["id"])
        by_file[name] = len(data["cases"])
        total += len(data["cases"])
    assert total == rh.EXPECTED_B_SEEDS == 45
    assert by_file["g11_direction"] >= 2 and by_file["g10_grounding"] >= 6 and by_file["g6_prescription"] >= 4
    assert sum(1 for i in notes if i.startswith("g10_fabricated_quote")) >= 3
    assert sum(1 for i in notes if i.startswith("g3_korean_numeral")) >= 3
    assert sum(1 for i in notes if i.startswith("g3_korean_numeral")) >= 2
    assert sum(1 for i in notes if i.startswith("g3_arabic")) >= 2
    assert sum(1 for i in notes if i.startswith("g10_no_grounding")) >= 2
    assert sum(1 for i in notes if i.startswith("g10_scale_mismatch")) >= 2
    assert sum(1 for i in notes if i in ("g5_question_not_interrogative", "g5_question_too_long",
                                          "g5_observation_question_form")) >= 2

    manifest = json.loads(
        (rh.FIXTURES_DIR / "a1_adversarial.json").read_text(encoding="utf-8"))
    assert len(manifest["seeded_violations"]) == rh.EXPECTED_PIPELINE_SEEDS == 10


def test_all_seeds_detected():
    """B축 시드 전수가 검출된다 (규칙별 파일 11종 + 우회)."""
    hit, total, rows, fails = rh.run_seeded_check()
    assert fails == []
    assert hit == total == rh.EXPECTED_B_SEEDS
    assert len(rows) == len(rh.SEEDED_FILE_ORDER) == 12


def test_pipeline_uses_single_prep_call_per_profile():
    """프로파일당 LLM 호출 1회(prep), 블록 2개 (ADR 0010). A1은 관찰 블록이 재생성 2회 소진 후 폴백된다."""
    client = MockLLMClient()
    profile = load_profile(ROOT / "data/profiles/p2_partial_borderline.json")
    results = generate_all(profile, client)
    assert list(results) == ["prep"] and results["prep"].block_count == 2
    assert [c["task"] for c in client.calls] == ["prep"]
    a1 = load_profile(ROOT / "data/profiles/a1_adversarial.json")
    r = generate_all(a1, MockLLMClient())["prep"]
    assert r.regen_count == 2 and r.fallback_blocks == ["observation_points"]
