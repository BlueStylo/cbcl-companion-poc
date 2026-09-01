"""CLI 진입점.

사용 예:
    python main.py --profile data/profiles/p2_partial_borderline.json --mock
    python main.py --compare data/profiles/p5a_paired_notes.json data/profiles/p5b_paired_notes.json --mock
    python main.py --profile ... --api   # .env의 LLM_* 변수 필요
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.compare_html import build_compare_html
from src.generator import generate_all
from src.guardrails import detect_crisis_signals
from src.llm_client import LLMError, make_client
from src.parser import ProfileError, load_profile
from src.report_html import build_crisis_html, build_report_html


def llm_failure_types() -> tuple:
    """fail-closed로 처리할 LLM 실패 계열 예외 목록.

    openai SDK가 설치된 경우 API 오류(연결 실패 포함)도 포함한다.
    """
    types: list[type] = [LLMError, ConnectionError, TimeoutError]
    try:
        import openai
        types.append(openai.APIError)
    except ImportError:
        pass
    return tuple(types)


def load_env_file(path: Path = Path(".env")) -> None:
    """.env의 KEY=VALUE를 환경 변수로 올린다 (이미 설정된 값은 유지)."""
    import os
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def summarize(pid: str, results: dict) -> str:
    regen = sum(r.regen_count for r in results.values())
    fallback = sum(len(r.fallback_blocks) for r in results.values())
    return f"{pid}: 가드레일 재생성 {regen}회, 안전 문구 대체 {fallback}건"


def main() -> int:
    ap = argparse.ArgumentParser(description="CBCL 보고서 동반 가이드 PoC")
    ap.add_argument("--profile", help="프로파일 JSON 1건 → 해설 리포트 HTML")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="동점 페어 2건 → 나란히 비교 HTML")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", default=True,
                      help="fixture 응답으로 오프라인 실행 (기본값)")
    mode.add_argument("--api", dest="mock", action="store_false",
                      help="실 LLM 호출 (.env의 LLM_* 필요)")
    ap.add_argument("--out", default="out", help="출력 디렉토리 (기본 out/)")
    args = ap.parse_args()

    if not args.profile and not args.compare:
        ap.error("--profile 또는 --compare 중 하나는 필요합니다")

    mode_label = "mock" if args.mock else "api"
    if not args.mock:
        load_env_file()
    client = make_client(mode_label)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 산출물은 모든 생성이 성공한 뒤에만 기록한다 (부분 산출물 방지, fail-closed)
    pending: list[tuple[Path, str]] = []
    notes: list[str] = []

    def queue_crisis(profile) -> None:
        """위기 신호 검출 시: LLM 호출 없이 상담 연결 안내만 큐에 넣는다."""
        path = out_dir / f"{profile.profile_id}.html"
        pending.append((path, build_crisis_html(profile)))
        notes.append(f"{profile.profile_id}: 위기 신호 검출 - 해설 생성을 중단하고 상담 연결 안내만 생성")
        notes.append(f"위기 안내 생성: {path}")

    compare_aborted = False
    try:
        if args.profile:
            profile = load_profile(args.profile)
            if detect_crisis_signals(profile):
                queue_crisis(profile)
            else:
                results = generate_all(profile, client)
                path = out_dir / f"{profile.profile_id}.html"
                pending.append((path, build_report_html(profile, results, mode_label)))
                notes.append(summarize(profile.profile_id, results))
                notes.append(f"리포트 생성: {path}")

        if args.compare:
            pa, pb = (load_profile(p) for p in args.compare)
            crisis = [p for p in (pa, pb) if detect_crisis_signals(p)]
            if crisis:
                for p in crisis:
                    queue_crisis(p)
                compare_aborted = True
            else:
                ra, rb = generate_all(pa, client), generate_all(pb, client)
                path = out_dir / f"compare_{pa.profile_id}_{pb.profile_id}.html"
                pending.append((path, build_compare_html(pa, ra, pb, rb, mode_label)))
                notes.append(summarize(pa.profile_id, ra))
                notes.append(summarize(pb.profile_id, rb))
                notes.append(f"비교 뷰 생성: {path}")
    except ProfileError as e:
        print("입력 프로파일 검증 실패 (리포트를 생성하지 않습니다):", file=sys.stderr)
        for msg in e.errors:
            print(f"  - {msg}", file=sys.stderr)
        return 1
    except llm_failure_types() as e:
        print(f"LLM 호출 실패: {e}", file=sys.stderr)
        print("생성이 완료되지 않아 리포트를 출력하지 않습니다.", file=sys.stderr)
        return 1

    for path, content in pending:
        path.write_text(content, encoding="utf-8")
    for line in notes:
        print(line)
    if compare_aborted:
        print("위기 신호가 검출된 프로파일이 있어 비교 뷰는 생성하지 않습니다.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
