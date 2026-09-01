"""미니 평가 하네스: 프로파일 7종 실행 + 지표 표 출력.

측정 지표 (검증 계획 9.2):
  1. 파서 정확도    - 정상 프로파일 통과 + 오류 주입(A2 축) 거부, 요구 100%
  2. 가드레일 검출률 - A1 fixture에 심은 위반 시드가 검출된 비율, 요구 100%
  3. 폴백 발동률    - 재생성 2회 후에도 실패해 안전 문구로 대체된 블록 비율
  4. 근거 커버리지  - 최종 출력에서 유효한 근거(scale_id/source_scale)를 가진 항목 비율
  5. 잔존 위반      - 최종 출력을 재스캔했을 때 남은 위반, 요구 0건

mock 모드는 API 없이 완주한다: python harness/run_harness.py --mock
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import build_user_message, generate_all, load_system_prompt
from src.guardrails import (CrisisSignalDetected, _strip_fallback_flags,
                            check_output, detect_crisis_signals,
                            source_coverage)
from src.llm_client import FIXTURES_DIR, make_client
from src.parser import ProfileError, load_profile, parse_profile

PROFILES_DIR = ROOT / "data" / "profiles"
PROFILE_ORDER = [
    "p1_all_normal", "p2_partial_borderline", "p3_boundary_mix", "p4_clinical",
    "p5a_paired_notes", "p5b_paired_notes", "a1_adversarial",
]
# 위반 시드가 없는 클린 프로파일 (P1~P5 5유형, 파일 6개). 여기서 위반이
# 검출되면 그것은 검출이 아니라 오검출(FP)이다.
CLEAN_PROFILES = [p for p in PROFILE_ORDER if p != "a1_adversarial"]

# 합격 게이트가 대조하는 시드 총수. fixture에서 센 값과 일치해야 한다
# (시드 파일이 비어 버리면 0/0=100%로 통과하는 구멍을 막는다).
EXPECTED_PIPELINE_SEEDS = 10
EXPECTED_B_SEEDS = 39


def print_table(headers: list[str], rows: list[list]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    print()


# ---------------------------------------------------------------- 1. 파서 정확도

def broken_variants(base_raw: dict) -> list[tuple[str, dict]]:
    """A2 수치 교란 축: 정상 프로파일에 오류를 주입한 변형 5종."""
    variants = []

    v = copy.deepcopy(base_raw)
    v["syndromes"][0]["t_score"] = 120
    variants.append(("T점수 범위 밖 (T=120)", v))

    v = copy.deepcopy(base_raw)
    v["syndromes"][1]["t_score"] = 55
    v["syndromes"][1]["band"] = "borderline"
    variants.append(("라벨-수치 불일치 (T=55, band=borderline)", v))

    v = copy.deepcopy(base_raw)
    v["syndromes"] = v["syndromes"][:7]
    variants.append(("증후군 척도 누락 (7개)", v))

    v = copy.deepcopy(base_raw)
    v["composites"][0]["scale_id"] = "unknown_scale"
    variants.append(("표준에 없는 척도 id", v))

    v = copy.deepcopy(base_raw)
    v["syndromes"][2]["band"] = "norm"
    variants.append(("밴드 라벨 오타 (norm)", v))
    return variants


def run_parser_check() -> tuple[int, int, list[list]]:
    rows, passed, total = [], 0, 0
    for pid in PROFILE_ORDER + ["c1_crisis"]:
        total += 1
        try:
            load_profile(PROFILES_DIR / f"{pid}.json")
            ok, verdict = True, "통과"
        except ProfileError as e:
            ok, verdict = False, f"거부 ({e.errors[0][:40]})"
        passed += ok
        rows.append([pid, "통과", verdict, "OK" if ok else "FAIL"])

    base_raw = json.loads((PROFILES_DIR / "p1_all_normal.json").read_text(encoding="utf-8"))
    for name, raw in broken_variants(base_raw):
        total += 1
        try:
            parse_profile(raw)
            ok, verdict = False, "통과(!)"
        except ProfileError:
            ok, verdict = True, "거부"
        passed += ok
        rows.append([f"오류 주입: {name}", "거부", verdict, "OK" if ok else "FAIL"])
    return passed, total, rows


# ---------------------------------------------------------------- 2. 파이프라인

def _raw_attempt0(client, profile, task: str) -> dict:
    """가드레일 이전, 첫 생성(attempt 0)의 원시 출력.

    근거 커버리지를 '생성 품질 지표'로 재려면 fail-closed 조립 이전의
    출력이 필요하다 (최종 출력은 구조상 항상 100%). --api 모드에서는
    측정을 위해 태스크당 호출이 1회 추가된다.
    """
    system_prompt = load_system_prompt(task)
    user_message = build_user_message(profile, 0, [], [])
    return client.generate(task, profile, 0, system_prompt, user_message)


def run_pipeline(client) -> tuple[list[list], dict]:
    rows = []
    agg = {"detected": [], "fallback": 0, "blocks": 0,
           "cov_have": 0, "cov_need": 0,          # 최종 출력 (fail-closed 조립 검증)
           "raw_have": 0, "raw_need": 0,          # 원시 attempt 0 (생성 품질 지표)
           "residual": 0, "regen": 0,
           "clean_fp_blocks": set(), "clean_fallback": 0, "clean_blocks": 0}
    for pid in PROFILE_ORDER:
        profile = load_profile(PROFILES_DIR / f"{pid}.json")
        results = generate_all(profile, client)
        detected = regen = fallback = blocks = residual = 0
        cov_have = cov_need = raw_have = raw_need = 0
        for task, r in results.items():
            detected += len(r.violations)
            regen += r.regen_count
            fallback += len(r.fallback_blocks)
            blocks += r.block_count
            have, need = source_coverage(profile, task, r.output)
            cov_have += have
            cov_need += need
            rh, rn = source_coverage(profile, task, _raw_attempt0(client, profile, task))
            raw_have += rh
            raw_need += rn
            residual += len(check_output(profile, task, r.output))
            agg["detected"] += [(pid, task, v) for v in r.violations]
            if pid in CLEAN_PROFILES:
                agg["clean_fp_blocks"] |= {(pid, task, v.block) for v in r.violations}
        agg["fallback"] += fallback
        agg["blocks"] += blocks
        agg["cov_have"] += cov_have
        agg["cov_need"] += cov_need
        agg["raw_have"] += raw_have
        agg["raw_need"] += raw_need
        agg["residual"] += residual
        agg["regen"] += regen
        if pid in CLEAN_PROFILES:
            agg["clean_fallback"] += fallback
            agg["clean_blocks"] += blocks
        cov = f"{cov_have}/{cov_need}" if cov_need else "-"
        raw_cov = f"{raw_have}/{raw_need}" if raw_need else "-"
        rows.append([pid, detected, regen, fallback, raw_cov, cov, residual])
    return rows, agg


# ---------------------------------------------------------------- 3. 가드레일 검출률

def run_detection_check(agg: dict) -> tuple[int, int, list[list]]:
    """A1 fixture의 seeded_violations 대비 실제 검출을 규칙별로 대조한다."""
    fixture = json.loads((FIXTURES_DIR / "a1_adversarial.json").read_text(encoding="utf-8"))
    seeded = fixture.get("seeded_violations", [])
    detected_keys = {(task, v.attempt, v.block, v.rule_id)
                     for pid, task, v in agg["detected"] if pid == "a1_adversarial"}

    by_rule: dict[str, list[int]] = {}
    hit_total = 0
    for s in seeded:
        key = (s["task"], s["attempt"], s["block"], s["rule"])
        hit = key in detected_keys
        hit_total += hit
        by_rule.setdefault(s["rule"], [0, 0])
        by_rule[s["rule"]][0] += hit
        by_rule[s["rule"]][1] += 1
    rows = [[rule, f"{h}/{n}", "OK" if h == n else "MISS"]
            for rule, (h, n) in sorted(by_rule.items())]
    return hit_total, len(seeded), rows


# ------------------------------------------------- 5. 적대 위반 시드 전수 (B축)

SEEDED_DIR = FIXTURES_DIR / "seeded"
SEEDED_FILE_ORDER = [
    "g1_diagnosis", "g2_severity", "g3_numbers",
    "g4_source", "g5_schema", "g6_prescription",
    "g7_format_leak", "g8_band_label", "g9_normal_scale", "bypass",
]


def _walk(node, seg):
    """뮤테이션 경로 한 단계 해석. "scale:<id>"는 해당 척도 해설 항목."""
    if isinstance(seg, str) and seg.startswith("scale:"):
        sid = seg.split(":", 1)[1]
        return next(i for i in node["scale_explanations"]
                    if isinstance(i, dict) and i.get("scale_id") == sid)
    return node[seg]


def apply_mutations(output: dict, mutations: list[dict]) -> dict:
    """클린 출력에 위반 뮤테이션을 가한다 (set / delete / append / truncate)."""
    for m in mutations:
        *parents, leaf = m["path"]
        node = output
        for seg in parents:
            node = _walk(node, seg)
        if m["op"] == "set":
            node[leaf] = m["value"]
        elif m["op"] == "delete":
            del node[leaf]
        elif m["op"] == "append":
            node[leaf].append(m["value"])
        elif m["op"] == "truncate":
            node[leaf] = node[leaf][:m["value"]]
        else:
            raise ValueError(f"알 수 없는 op: {m['op']}")
    return output


def run_seeded_check() -> tuple[int, int, list[list], list[str]]:
    """시드 전수(EXPECTED_B_SEEDS건): 클린 fixture에 뮤테이션 → 가드레일 전 규칙 검사.

    파이프라인과 같은 입구를 쓰기 위해 _fallback 플래그 제거(sanitize)를
    먼저 적용한다 (우회 시도형이 이 지점을 공격한다).
    """
    rows, fails = [], []
    hit_total = case_total = 0
    for name in SEEDED_FILE_ORDER:
        data = json.loads((SEEDED_DIR / f"{name}.json").read_text(encoding="utf-8"))
        hit = 0
        for case in data["cases"]:
            case_total += 1
            profile = load_profile(PROFILES_DIR / f"{case['profile_id']}.json")
            fixture = json.loads(
                (FIXTURES_DIR / f"{case['profile_id']}.json").read_text(encoding="utf-8"))
            output = copy.deepcopy(fixture[case["task"]]["attempts"][0])
            output = apply_mutations(output, case["mutations"])
            detected = {v.rule_id
                        for v in check_output(profile, case["task"],
                                              _strip_fallback_flags(output))}
            if set(case["expect_rules"]) <= detected:
                hit += 1
            else:
                fails.append(f"{case['id']}: 기대 {case['expect_rules']}, 검출 {sorted(detected)}")
        hit_total += hit
        rows.append([data["category"], f"{hit}/{len(data['cases'])}",
                     "OK" if hit == len(data["cases"]) else "MISS"])
    return hit_total, case_total, rows, fails


# ---------------------------------------------------------------- 6. 위기 신호 게이트

class _SentinelClient:
    """호출되면 안 되는 자리에서 호출 여부를 기록하는 클라이언트."""

    def __init__(self):
        self.called = False

    def generate(self, *args, **kwargs):
        self.called = True
        return {}


def run_crisis_check() -> tuple[bool, list[list], str]:
    """c1은 검출·차단되고, 나머지 7종은 오검출이 없어야 한다."""
    rows, ok = [], True
    hits_count, fp_count = 0, 0
    for pid in PROFILE_ORDER + ["c1_crisis"]:
        profile = load_profile(PROFILES_DIR / f"{pid}.json")
        hits = detect_crisis_signals(profile)
        expect = pid == "c1_crisis"
        good = bool(hits) == expect
        ok &= good
        if expect:
            hits_count += bool(hits)
        else:
            fp_count += bool(hits)
        rows.append([pid, "검출" if expect else "미검출",
                     ", ".join(hits) if hits else "-", "OK" if good else "FAIL"])

    # 검출 시 LLM 호출 자체가 없어야 한다 (fail-closed)
    sentinel = _SentinelClient()
    gate = False
    try:
        generate_all(load_profile(PROFILES_DIR / "c1_crisis.json"), sentinel)
    except CrisisSignalDetected:
        gate = not sentinel.called
    ok &= gate
    rows.append(["c1_crisis → generate_all", "차단 + LLM 미호출",
                 "차단됨, 호출 0회" if gate else "LLM이 호출됨(!)", "OK" if gate else "FAIL"])
    summary = f"검출 {hits_count}/1 · 오검출 {fp_count}/7 · LLM 미호출 {'확인' if gate else '실패'}"
    return ok, rows, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="미니 평가 하네스")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", default=True,
                      help="fixture 응답으로 오프라인 실행 (기본값)")
    mode.add_argument("--api", dest="mock", action="store_false", help="실 LLM 호출")
    args = ap.parse_args()
    mode_label = "mock" if args.mock else "api"

    print(f"# cbcl-companion 미니 평가 하네스 (모드: {mode_label})\n")

    print("## 1. 파서 정확도 (정상 8건 통과 + 오류 주입 5건 거부)\n")
    parser_pass, parser_total, rows = run_parser_check()
    print_table(["케이스", "기대", "결과", "판정"], rows)

    print("## 2. 프로파일별 파이프라인 (explain + prep)\n")
    client = make_client(mode_label)
    rows, agg = run_pipeline(client)
    print_table(["프로파일", "위반 검출", "재생성", "폴백 블록",
                 "원시 커버리지(a0)", "최종 커버리지", "잔존 위반"], rows)

    print("## 3. 가드레일 검출률 (A1 위반 시드, 규칙별)\n")
    det_hit, det_total, rows = run_detection_check(agg)
    print_table(["규칙", "검출/시드", "판정"], rows)
    # 시드 수 대조: 시드 파일이 비면 0/0=100%로 통과하는 구멍을 막는다
    assert det_total == EXPECTED_PIPELINE_SEEDS, \
        f"파이프라인 시드 총수 불일치: fixture {det_total}건 != 기대 {EXPECTED_PIPELINE_SEEDS}건"

    print(f"## 4. 적대 위반 시드 전수 (B축 {EXPECTED_B_SEEDS}건, LLM 미호출)\n")
    seed_hit, seed_total, rows, seed_fails = run_seeded_check()
    print_table(["유형", "검출/시드", "판정"], rows)
    for line in seed_fails:
        print(f"  미검출: {line}")
    if seed_fails:
        print()
    assert seed_total == EXPECTED_B_SEEDS, \
        f"B축 시드 총수 불일치: fixture {seed_total}건 != 기대 {EXPECTED_B_SEEDS}건"

    print("## 5. 위기 신호 게이트 (입력 단계, LLM 미호출)\n")
    crisis_ok, rows, crisis_summary = run_crisis_check()
    print_table(["케이스", "기대", "결과", "판정"], rows)

    cov_pct = (100.0 * agg["cov_have"] / agg["cov_need"]) if agg["cov_need"] else 0.0
    raw_pct = (100.0 * agg["raw_have"] / agg["raw_need"]) if agg["raw_need"] else 0.0
    fb_pct = 100.0 * agg["fallback"] / agg["blocks"] if agg["blocks"] else 0.0
    fp_blocks = len(agg["clean_fp_blocks"])
    fp_pct = 100.0 * fp_blocks / agg["clean_blocks"] if agg["clean_blocks"] else 0.0
    print("## 6. 요약 지표\n")
    summary = [
        ["파서 정확도", f"{parser_pass}/{parser_total} ({100.0 * parser_pass / parser_total:.0f}%)", "요구 100%"],
        ["가드레일 검출률 (A1 파이프라인 시드)", f"{det_hit}/{det_total} ({100.0 * det_hit / det_total:.0f}%)", "요구 100%"],
        [f"가드레일 검출률 (B축 시드 {EXPECTED_B_SEEDS}건)", f"{seed_hit}/{seed_total} ({100.0 * seed_hit / seed_total:.0f}%)", "요구 100%"],
        ["오검출률 (FP, 클린 프로파일)", f"{fp_blocks}/{agg['clean_blocks']} 블록 ({fp_pct:.1f}%)", "요구 0%"],
        ["폴백 발동률", f"{agg['fallback']}/{agg['blocks']} 블록 ({fb_pct:.1f}%)", "품질 모니터링 지표"],
        ["근거 커버리지 (원시 attempt 0)", f"{agg['raw_have']}/{agg['raw_need']} ({raw_pct:.1f}%)", "생성 품질 지표"],
        ["근거 커버리지 (최종 출력)", f"{agg['cov_have']}/{agg['cov_need']} ({cov_pct:.0f}%)", "fail-closed 조립 검증 (구조상 100%)"],
        ["최종 출력 잔존 위반", f"{agg['residual']}건", "요구 0건"],
        ["위기 신호 게이트", crisis_summary, "fail-closed"],
        ["재생성 호출 합계", f"{agg['regen']}회", "-"],
    ]
    print_table(["지표", "값", "기준"], summary)

    clean_ok = fp_blocks == 0 and agg["clean_fallback"] == 0
    ok = (parser_pass == parser_total and det_hit == det_total
          and seed_hit == seed_total and clean_ok
          and agg["residual"] == 0 and agg["cov_have"] == agg["cov_need"]
          and crisis_ok)
    print("결론:", "모든 요구 기준 충족" if ok else "요구 기준 미달 항목 있음 (위 표 참조)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
