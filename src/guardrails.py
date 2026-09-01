"""출력 안전성 가드레일 (규칙 엔진, LLM 미사용).

출력 규칙 6종:
  G1 진단명 사전         - 진단명 등장 자체를 위반 처리 (부정문 포함, 의도된 과검출)
  G2 심각성 단정(양방향) - 심각 쪽 단정과 근거 없는 낙관 보증을 모두 차단
  G3 수치 대조           - 본문 수치를 입력 수치 집합과 대조 + 에코 필드(t_score, band) 대조
  G4 근거 링크           - scale_id / source_scale 이 입력의 실제 척도와 매칭되는지
  G5 스키마              - 출력 JSON 구조, 필수 필드, 항목 수
  G6 처방·치료 권고      - 약물, 치료 시작, 의료기관 방문 지시 차단
                           (허용 형태는 "예약된 상담에서 상담사와 이야기해 보세요" 하나뿐)

위반 블록은 최대 2회 재생성하고, 그래도 실패하면 사전 작성 안전 문구로
대체한다 (fail-closed). 리포트가 아예 안 나가는 일은 없고, 검증 안 된
문장이 나가는 일도 없다.

입력 게이트 1종: 위기 신호 검출 (detect_crisis_signals). 보호자 의견에
긴급 키워드가 있으면 LLM 호출 자체를 하지 않는다 (generator가 이 함수로
차단하고, 화면은 상담 연결 안내만 출력한다). 사전은 보수적으로 과검출을
허용한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parser import (BAND_KO, COMPOSITE_IDS, SCALE_DEFINITIONS, SYNDROME_IDS,
                     CBCLProfile)

MAX_REGEN = 2  # 첫 생성 이후 블록 단위 재생성 횟수

SAFE_SCALE_TEXT = "이 척도에 대한 자세한 설명은 예약된 상담에서 상담사에게 직접 들으시길 권합니다."
SAFE_GENERIC_TEXT = "이 부분의 자동 생성 문구는 검증을 통과하지 못했습니다. 예약된 상담에서 상담사에게 직접 들으시길 권합니다."
SAFE_LIMITS_TEXT = (
    "이 검사는 보호자의 보고를 바탕으로 한 선별 도구이며, 단일 검사만으로는 "
    "어떤 것도 확정되지 않습니다. 결과의 해석은 예약된 상담에서 상담사와 이야기해 보세요."
)
SAFE_QUESTION = "이번 결과에서 무엇부터 살펴보면 좋을지, 예약된 상담에서 상담사에게 직접 여쭤보시길 권합니다."
SAFE_OBSERVATION = "상담 전까지 아이의 하루 중 인상 깊었던 장면을 하루 한 줄로 적어 두시면 상담에서 쓸 수 있습니다."
SAFE_BRIEFING = "자동 사전 요약 생성이 검증을 통과하지 못했습니다. 원 결과지의 수치를 직접 확인해 주세요."

# --- G1: 진단명 사전 (부정문·완곡형 포함, 등장 자체를 위반 처리) ---
DIAGNOSIS_PATTERN = re.compile(
    r"(ADHD|주의력\s*결핍|자폐|우울증|불안\s*장애|틱\s*장애|품행\s*장애"
    r"|발달\s*장애|학습\s*장애|조현병|강박\s*장애|반항\s*장애|공황\s*장애)",
    re.IGNORECASE,
)

# --- G2: 심각성 단정 (양방향) ---
SEVERITY_PATTERNS = [re.compile(p) for p in (
    r"심각",
    r"위험(?:한|이|합|해)",
    r"장애가\s*있",
    r"문제가\s*(?:크|많|심)",
    r"우려스러운",
    r"병적",
)]
OPTIMISM_PATTERNS = [re.compile(p) for p in (
    r"걱정\s*(?:안\s*하|하지\s*않|마세요)",
    r"아무\s*문제\s*없",
    r"괜찮(?:습니다|아질|을)",
    r"안심하",
    r"좋아질\s*(?:겁니다|거예요|것입니다)",
    r"문제\s*없습니다",
)]

# --- G6: 처방·치료 권고 (설계 문서 R2) ---
PRESCRIPTION_PATTERNS = [re.compile(p) for p in (
    r"약물",
    r"약을\s*(?:복용|먹)",
    r"치료(?:를|가)?\s*(?:받|필요|시작)",
    r"치료\s*프로그램",
    r"병원(?:에|을)?\s*(?:가|방문)",
    r"처방",
    r"의료\s*기관",
    # 완곡한 의뢰 시사도 차단: "전문의 상담이 필요한 수준으로 보입니다" 류.
    # 허용된 전문가 안내 형태는 "예약된 상담에서 상담사와..." 하나뿐이다.
    r"전문의",
    r"정신과",
    r"소아\s*정신",
)]

# --- G3: 본문 수치 추출 패턴 ---
NUMBER_PATTERNS = [re.compile(p) for p in (
    r"[Tt]\s*=\s*(\d{1,3})",
    r"[Tt]\s*점수\s*(\d{1,3})",
    r"(\d{1,3})\s*T",
    r"(\d{1,3})\s*점(?!검)",
    r"(\d{1,3})\s*백분위",
    r"(\d{1,3})\s*%",
)]


@dataclass
class Violation:
    rule_id: str          # "G1".."G5"
    block: str            # 위반이 발견된 블록 id (예: "scale:attention")
    matched: str          # 매칭된 문자열 또는 불일치 값 쌍
    attempt: int = -1     # 몇 번째 생성에서 발견됐는지 (run_with_guardrails가 기록)


@dataclass
class SafeResult:
    """가드레일 루프를 통과한 최종 결과."""
    task: str
    output: dict
    violations: list[Violation] = field(default_factory=list)
    regen_count: int = 0                       # 재생성 호출 횟수 (첫 생성 제외)
    fallback_blocks: list[str] = field(default_factory=list)
    block_count: int = 0


# ---------------------------------------------------------------- 위기 신호 게이트 (입력 단계)

# 보수적 사전: 과검출을 허용한다 (fail-closed). 미검출 1건의 비용이
# 오검출 여러 건의 비용보다 크다.
CRISIS_PATTERNS = [re.compile(p) for p in (
    r"자해",
    r"자살",
    r"죽고\s*싶",
    r"죽고싶",
    r"죽어\s*버리",
    r"죽었으면",
    r"극단적\s*선택",
    r"살고\s*싶지\s*않",
    r"살기\s*싫",
    r"사라지고\s*싶",
    r"사라져\s*버리",
    r"목숨",
    r"스스로\s*(?:를\s*)?(?:해치|다치|상처)",
    r"몸에\s*상처",
    r"상처를\s*(?:내|냈|낸)",
)]


class CrisisSignalDetected(RuntimeError):
    """입력에서 위기 신호가 검출됨. LLM 호출 없이 상담 연결 안내만 출력한다."""

    def __init__(self, keywords: list[str]):
        self.keywords = keywords
        super().__init__(f"위기 신호 검출: {keywords}")


def detect_crisis_signals(profile: CBCLProfile) -> list[str]:
    """보호자 의견 텍스트에서 긴급 키워드를 찾는다 (매칭 문자열 목록 반환).

    비어 있지 않으면 파이프라인은 해설 생성을 시작하지 않아야 한다.
    """
    found: list[str] = []
    for note in profile.caregiver_notes:
        for pat in CRISIS_PATTERNS:
            m = pat.search(note)
            if m and m.group(0) not in found:
                found.append(m.group(0))
    return found


# ---------------------------------------------------------------- 블록 분해

def expected_blocks(profile: CBCLProfile, task: str) -> list[str]:
    """이 task에서 반드시 채워져야 하는 블록 id 목록."""
    if task == "explain":
        return (["overview"]
                + [f"scale:{sid}" for sid in (*COMPOSITE_IDS, *SYNDROME_IDS)]
                + ["limits", "before_counseling"])
    return ["questions_for_counselor", "observation_points", "counselor_briefing"]


def split_blocks(profile: CBCLProfile, task: str, raw: dict) -> dict[str, object]:
    """출력 JSON을 블록 단위로 나눈다 (상위 스키마 통과 후에 호출)."""
    if task == "explain":
        blocks: dict[str, object] = {
            "overview": raw.get("overview"),
            "limits": raw.get("limits"),
            "before_counseling": raw.get("before_counseling"),
        }
        for item in raw.get("scale_explanations", []):
            sid = item.get("scale_id") if isinstance(item, dict) else None
            blocks[f"scale:{sid}"] = item
        return blocks
    return {k: raw.get(k) for k in expected_blocks(profile, task)}


# ---------------------------------------------------------------- 검사기

def _check_text(block: str, text: str, profile: CBCLProfile) -> list[Violation]:
    """텍스트 1개에 대한 G1/G2/G3 검사."""
    found: list[Violation] = []
    m = DIAGNOSIS_PATTERN.search(text)
    if m:
        found.append(Violation("G1", block, m.group(0)))
    for pat in SEVERITY_PATTERNS + OPTIMISM_PATTERNS:
        m = pat.search(text)
        if m:
            found.append(Violation("G2", block, m.group(0)))
    for pat in PRESCRIPTION_PATTERNS:
        m = pat.search(text)
        if m:
            found.append(Violation("G6", block, m.group(0)))
    allowed = profile.allowed_numbers()
    for pat in NUMBER_PATTERNS:
        for m in pat.finditer(text):
            if int(m.group(1)) not in allowed:
                found.append(Violation("G3", block, m.group(0)))
    return found


def _require_str(block: str, value, name: str) -> list[Violation]:
    if not isinstance(value, str) or not value.strip():
        return [Violation("G5", block, f"{name}: 문자열 필수")]
    return []


def check_top_schema(task: str, raw) -> list[Violation]:
    """출력 최상위 구조 검사 (여기서 걸리면 전체 재생성)."""
    if not isinstance(raw, dict):
        return [Violation("G5", "*", "출력이 JSON 객체가 아님")]
    keys = (("overview", "scale_explanations", "limits", "before_counseling")
            if task == "explain"
            else ("questions_for_counselor", "observation_points", "counselor_briefing"))
    missing = [k for k in keys if k not in raw]
    if missing:
        return [Violation("G5", "*", f"필수 키 누락: {missing}")]
    if task == "explain" and not isinstance(raw["scale_explanations"], list):
        return [Violation("G5", "*", "scale_explanations는 배열이어야 함")]
    return []


def _check_scale_block(profile: CBCLProfile, block: str, item) -> list[Violation]:
    if not isinstance(item, dict):
        return [Violation("G5", block, "척도 해설은 객체여야 함")]
    found: list[Violation] = []
    sid = item.get("scale_id")
    scale = profile.scale_map().get(sid)
    if scale is None:
        return [Violation("G4", block, f"입력에 없는 scale_id: {sid!r}")]
    # 에코 필드 대조 (수치·판정 위변조 검출)
    if item.get("t_score") != scale.t_score:
        found.append(Violation("G3", block, f"t_score 에코 불일치: {item.get('t_score')!r} != {scale.t_score}"))
    if item.get("band") != scale.band:
        found.append(Violation("G3", block, f"band 에코 불일치: {item.get('band')!r} != {scale.band}"))
    for name in ("what_it_measures", "what_the_number_means", "everyday_example"):
        vs = _require_str(block, item.get(name), name)
        found += vs
        if not vs:
            found += _check_text(block, item[name], profile)
    return found


def _check_items_block(profile: CBCLProfile, block: str, items, text_key: str,
                       lo: int, hi: int) -> list[Violation]:
    """questions_for_counselor / observation_points 공용 검사."""
    if not isinstance(items, list):
        return [Violation("G5", block, "배열이어야 함")]
    real_items = [it for it in items if not (isinstance(it, dict) and it.get("_fallback"))]
    if real_items != items:  # 폴백 문구 블록은 재검사하지 않는다
        return []
    found: list[Violation] = []
    if not (lo <= len(items) <= hi):
        found.append(Violation("G5", block, f"항목 수 {len(items)}건 (요구: {lo}~{hi})"))
    valid_ids = set(profile.scale_map())
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            found.append(Violation("G5", f"{block}[{i}]", "항목은 객체여야 함"))
            continue
        vs = _require_str(f"{block}[{i}]", it.get(text_key), text_key)
        found += vs
        if not vs:
            found += [Violation(v.rule_id, block, v.matched)
                      for v in _check_text(block, it[text_key], profile)]
        if it.get("source_scale") not in valid_ids:
            found.append(Violation("G4", block, f"source_scale 매칭 실패: {it.get('source_scale')!r}"))
    return found


def check_block(profile: CBCLProfile, task: str, block: str, content) -> list[Violation]:
    """블록 1개에 대한 전체 규칙 검사."""
    if isinstance(content, dict) and content.get("_fallback"):
        return []  # 사전 작성 고정 문구
    if block.startswith("scale:"):
        return _check_scale_block(profile, block, content)
    if block == "questions_for_counselor":
        return _check_items_block(profile, block, content, "question", 5, 7)
    if block == "observation_points":
        return _check_items_block(profile, block, content, "point", 3, 5)
    # overview / limits / before_counseling / counselor_briefing: 순수 텍스트
    vs = _require_str(block, content, block)
    if vs:
        return vs
    return _check_text(block, content, profile)


def check_output(profile: CBCLProfile, task: str, raw) -> list[Violation]:
    """출력 전체 일괄 검사 (하네스의 seeded 검사와 최종 잔존 위반 스캔용)."""
    top = check_top_schema(task, raw)
    if top:
        return top
    found: list[Violation] = []
    blocks = split_blocks(profile, task, raw)
    for block, content in blocks.items():
        found += check_block(profile, task, block, content)
    for block in expected_blocks(profile, task):
        if block not in blocks:
            found.append(Violation("G5", block, "블록 누락"))
    return found


# ---------------------------------------------------------------- 폴백

def fallback_for(profile: CBCLProfile, task: str, block: str):
    """검증에 끝내 실패한 블록을 대체할 사전 작성 안전 문구 (fail-closed)."""
    if block.startswith("scale:"):
        sid = block.split(":", 1)[1]
        scale = profile.scale_map()[sid]
        return {
            "scale_id": sid, "t_score": scale.t_score, "band": scale.band,
            "what_it_measures": SCALE_DEFINITIONS[sid],
            "what_the_number_means": f"원 보고서에 표기된 라벨은 '{BAND_KO[scale.band]}'입니다. " + SAFE_SCALE_TEXT,
            "everyday_example": "",
            "_fallback": True,
        }
    if block == "questions_for_counselor":
        return [{"question": SAFE_QUESTION, "source_scale": "total_problems", "_fallback": True}]
    if block == "observation_points":
        return [{"point": SAFE_OBSERVATION, "source_scale": "total_problems", "_fallback": True}]
    if block == "counselor_briefing":
        return SAFE_BRIEFING
    if block == "limits":
        return SAFE_LIMITS_TEXT
    return SAFE_GENERIC_TEXT  # overview / before_counseling


def rebuild(profile: CBCLProfile, task: str, blocks: dict[str, object]) -> dict:
    """블록 딕셔너리를 출력 스키마 형태로 재조립한다."""
    if task == "explain":
        return {
            "overview": blocks["overview"],
            "scale_explanations": [blocks[f"scale:{sid}"]
                                   for sid in (*COMPOSITE_IDS, *SYNDROME_IDS)],
            "limits": blocks["limits"],
            "before_counseling": blocks["before_counseling"],
        }
    return {k: blocks[k] for k in expected_blocks(profile, task)}


def _strip_fallback_flags(value):
    """LLM 출력이 _fallback 플래그를 흉내 내 검사를 우회하지 못하게 제거한다.

    _fallback은 이 모듈이 폴백을 삽입할 때만 붙는 내부 표식이다.
    """
    if isinstance(value, dict):
        return {k: _strip_fallback_flags(v) for k, v in value.items() if k != "_fallback"}
    if isinstance(value, list):
        return [_strip_fallback_flags(v) for v in value]
    return value


# ---------------------------------------------------------------- 메인 루프

def run_with_guardrails(profile: CBCLProfile, task: str, generate_fn,
                        max_regen: int = MAX_REGEN) -> SafeResult:
    """생성 → 검사 → 위반 블록만 재생성(최대 max_regen회) → 폴백.

    generate_fn(attempt, pending_blocks, feedback_violations) -> dict
    """
    expected = expected_blocks(profile, task)
    final: dict[str, object] = {}
    log: list[Violation] = []
    regen_count = 0

    for attempt in range(max_regen + 1):
        if attempt > 0:
            regen_count += 1
        pending = [b for b in expected if b not in final]
        feedback = [v for v in log if v.attempt == attempt - 1]
        raw = _strip_fallback_flags(generate_fn(attempt, pending, feedback))

        top = check_top_schema(task, raw)
        if top:
            for v in top:
                v.attempt = attempt
            log += top
            continue

        blocks = split_blocks(profile, task, raw)
        # 입력에 없는 척도를 지어낸 경우: 내용은 버리되 위반으로 기록
        for block in blocks:
            if block.startswith("scale:") and block not in expected:
                log.append(Violation("G4", block, "입력에 없는 scale_id", attempt))
        for block in pending:
            if block not in blocks:
                log.append(Violation("G5", block, "블록 누락", attempt))
                continue
            vs = check_block(profile, task, block, blocks[block])
            if vs:
                for v in vs:
                    v.attempt = attempt
                log += vs
            else:
                final[block] = blocks[block]
        if len(final) == len(expected):
            break

    fallback_blocks = [b for b in expected if b not in final]
    for block in fallback_blocks:
        final[block] = fallback_for(profile, task, block)

    return SafeResult(task=task, output=rebuild(profile, task, final),
                      violations=log, regen_count=regen_count,
                      fallback_blocks=fallback_blocks, block_count=len(expected))


# ---------------------------------------------------------------- 지표 헬퍼

def source_coverage(profile: CBCLProfile, task: str, output: dict) -> tuple[int, int]:
    """근거 커버리지: (유효 근거를 가진 항목 수, 근거가 필요한 항목 수).

    폴백으로 대체된 항목은 분모에서 제외한다 (사전 작성 고정 문구).
    """
    valid_ids = set(profile.scale_map())
    have, need = 0, 0
    if task == "explain":
        for item in output.get("scale_explanations", []):
            if isinstance(item, dict) and item.get("_fallback"):
                continue
            need += 1
            if isinstance(item, dict) and item.get("scale_id") in valid_ids:
                have += 1
    else:
        for key in ("questions_for_counselor", "observation_points"):
            for item in output.get(key, []):
                if isinstance(item, dict) and item.get("_fallback"):
                    continue
                need += 1
                if isinstance(item, dict) and item.get("source_scale") in valid_ids:
                    have += 1
    return have, need
