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
    """set / delete / append / truncate와 scale:<id> 경로 해석."""
    out = {
        "overview": "이전",
        "scale_explanations": [{"scale_id": "attention", "x": 1}],
        "items": [1, 2, 3],
    }
    rh.apply_mutations(out, [
        {"op": "set", "path": ["overview"], "value": "이후"},
        {"op": "set", "path": ["scale:attention", "y"], "value": 9},
        {"op": "delete", "path": ["scale:attention", "x"]},
        {"op": "append", "path": ["items"], "value": 4},
        {"op": "truncate", "path": ["items"], "value": 2},
    ])
    assert out["overview"] == "이후"
    assert out["scale_explanations"][0] == {"scale_id": "attention", "y": 9}
    assert out["items"] == [1, 2]


def test_undetectable_seed_counted_as_miss(tmp_path, monkeypatch):
    """검출될 수 없는 기대 규칙(뮤테이션 없음 + G1 기대)은 MISS로 집계."""
    case_file = {"category": "테스트", "cases": [{
        "id": "impossible", "task": "explain",
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

    fail-closed 조립이라 구조상 항상 100%임을 A1로 확인한다.
    """
    profile = load_profile(ROOT / "data/profiles/a1_adversarial.json")
    client = MockLLMClient()

    raw = rh._raw_attempt0(client, profile, "prep")
    raw_have, raw_need = source_coverage(profile, "prep", raw)
    assert raw_have < raw_need  # 원시 출력에는 근거 위반 시드가 있다

    final = generate_all(profile, client)["prep"].output
    final_have, final_need = source_coverage(profile, "prep", final)
    assert final_have == final_need


def test_seed_inventory_matches_gate_constants():
    """시드 파일 재고: B축 30건, 파이프라인 10건, expect_rules 공백 없음."""
    total = 0
    for name in rh.SEEDED_FILE_ORDER:
        data = json.loads(
            (rh.SEEDED_DIR / f"{name}.json").read_text(encoding="utf-8"))
        for case in data["cases"]:
            assert case["expect_rules"], f"{case['id']}: expect_rules 비어 있음"
        total += len(data["cases"])
    assert total == rh.EXPECTED_B_SEEDS == 30

    manifest = json.loads(
        (rh.FIXTURES_DIR / "a1_adversarial.json").read_text(encoding="utf-8"))
    assert len(manifest["seeded_violations"]) == rh.EXPECTED_PIPELINE_SEEDS == 10
