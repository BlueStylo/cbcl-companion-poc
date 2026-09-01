"""해설(기능 1)과 상담 준비(기능 2) 생성.

프롬프트 계약의 정본은 prompts/ 아래 파일이다. LLM에는 파서가 검증한
구조화 JSON만 넘기고(아동 이름은 제외 - 식별자 마스킹), 출력은
guardrails의 블록 단위 재생성 루프를 거친 SafeResult로 돌려준다.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import guardrails
from .parser import CBCLProfile

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
TASK_PROMPT_FILES = {"explain": "explainer_system.md", "prep": "counsel_prep_system.md"}


def load_system_prompt(task: str) -> str:
    """프롬프트 계약 전문을 파일에서 읽는다 (코드 밖 정본)."""
    return (PROMPTS_DIR / TASK_PROMPT_FILES[task]).read_text(encoding="utf-8")


def profile_payload(profile: CBCLProfile) -> dict:
    """LLM에 넘기는 최소 구조화 입력. 아동 이름(alias)은 넘기지 않는다."""
    scale = lambda s: {"scale_id": s.scale_id, "name_ko": s.name_ko,
                       "t_score": s.t_score, "band": s.band}
    return {
        "instrument": profile.instrument,
        "child": {"sex": profile.child.sex, "age_years": profile.child.age_years,
                  "age_months": profile.child.age_months,
                  "norm_group": profile.child.norm_group},
        "composites": [scale(s) for s in profile.composites],
        "syndromes": [scale(s) for s in profile.syndromes],
        "caregiver_notes": list(profile.caregiver_notes),
        "counseling_scheduled": profile.counseling_scheduled,
        "days_until_counseling": profile.days_until_counseling,
    }


def build_user_message(profile: CBCLProfile, attempt: int,
                       pending: list[str], feedback: list) -> str:
    """사용자 메시지 조립. 재생성 시 위반 피드백을 앞에 붙인다."""
    body = json.dumps(profile_payload(profile), ensure_ascii=False, indent=2)
    if attempt == 0:
        return body
    lines = [f"- 블록 {v.block}: 규칙 {v.rule_id} 위반 ({v.matched})" for v in feedback]
    return (
        "이전 출력에서 아래 위반이 발견되어 재생성합니다. 같은 출력 스키마 전체를 "
        "다시 작성하되, 다음 블록의 위반을 반드시 제거하세요.\n"
        + "\n".join(lines)
        + "\n\n검사 결과 JSON:\n" + body
    )


def generate_safe(profile: CBCLProfile, task: str, client) -> guardrails.SafeResult:
    """생성 1건을 가드레일 루프(재생성 2회 + 폴백)로 감싸 실행한다.

    위기 신호가 검출된 입력은 LLM 호출 자체를 하지 않는다 (입력 게이트).
    """
    crisis = guardrails.detect_crisis_signals(profile)
    if crisis:
        raise guardrails.CrisisSignalDetected(crisis)
    system_prompt = load_system_prompt(task)

    def gen_fn(attempt: int, pending: list[str], feedback: list) -> dict:
        user_message = build_user_message(profile, attempt, pending, feedback)
        return client.generate(task, profile, attempt, system_prompt, user_message)

    return guardrails.run_with_guardrails(profile, task, gen_fn)


def generate_all(profile: CBCLProfile, client) -> dict[str, guardrails.SafeResult]:
    """기능 1(해설)과 기능 2(상담 준비)를 모두 생성한다."""
    return {task: generate_safe(profile, task, client) for task in ("explain", "prep")}
