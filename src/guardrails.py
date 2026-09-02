"""출력 안전성 가드레일 (규칙 엔진, LLM 미사용).

출력 규칙 9종:
  G1 진단명 사전         - 진단명 등장 자체를 위반 처리 (부정문 포함, 의도된 과검출)
  G2 심각성 단정(양방향) - 심각 쪽 단정과 근거 없는 낙관 보증을 모두 차단
  G3 수치 대조           - 본문 수치를 입력 수치 집합과 대조
  G4 근거 링크           - 질문·관찰 포인트의 source_scale 이 입력의 실제 척도와 매칭되는지
  G5 스키마              - 출력 JSON 구조, 필수 필드, 항목 수
  G6 처방·치료 권고      - 약물, 치료 시작, 의료기관 방문 지시 차단
                           (허용 형태는 "예약된 상담에서 상담사와 이야기해 보세요" 하나뿐)
  G7 형식 누출           - 보호자 노출 텍스트에 scale_id 값, 영문 소문자 식별자, "scale_id",
                           괄호 안 영문 코드가 새면 위반 (상담사용 사전 요약은 대상 아님)
  G8 밴드 라벨 정합      - 밴드 어휘는 보고서 라벨(정상/준임상/임상)만 허용하고, 언급된
                           척도(척도명 사전으로 매핑)의 실제 band와 일치해야 함.
                           "경계 수준", "경계성", "borderline", "위험군", "높은 편" 등은 위반
  G9 정상 척도 근거 금지 - 질문·관찰 포인트의 source_scale band가 normal이면 위반
                           (종합지표 포함). 정상 척도는 해설 카드에서만 다룬다.
                           전 척도 정상 프로파일은 앵커가 없으므로 적용하지 않는다.
  G10 예시 오염          - 프롬프트 작성 예시의 관찰 문구("학원 숙제", "놀이터" 등)가
                           이 프로파일의 caregiver_notes에 없는데 질문·관찰·요약에
                           등장하면 위반 (보호자가 하지 않은 말을 인용한 것)

위반 블록은 최대 2회 재생성하고, 그래도 실패하면 사전 작성 안전 문구로
대체한다 (fail-closed). 리포트가 아예 안 나가는 일은 없고, 검증 안 된
문장이 나가는 일도 없다.

LLM이 쓰는 블록은 5개뿐이다: explain의 overview(보호자 관찰과 소견을 잇는
연결 문단)·before_counseling, prep의 질문·관찰 포인트·상담사용 요약.
척도 카드 해설과 한계 고지는 고정 문구(scale_texts.py)라 검사 대상이 아니다.

입력 게이트 1종: 위기 신호 검출 (detect_crisis_signals). 보호자 의견에
긴급 키워드가 있으면 LLM 호출 자체를 하지 않는다 (generator가 이 함수로
차단하고, 화면은 상담 연결 안내만 출력한다). 사전은 보수적으로 과검출을
허용한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parser import BAND_KO, SCALE_NAMES, CBCLProfile

MAX_REGEN = 2  # 첫 생성 이후 블록 단위 재생성 횟수

SAFE_GENERIC_TEXT = "이 부분의 자동 생성 문구는 검증을 통과하지 못했습니다. 예약된 상담에서 상담사에게 직접 들으시길 권합니다."
SAFE_OVERVIEW = (
    "보호자의 관찰과 검사 소견을 잇는 자동 요약이 검증을 통과하지 못했습니다. "
    "각 척도의 라벨은 아래 카드에 원 보고서 그대로 표기되어 있으며, 그 의미는 예약된 상담에서 상담사와 이야기해 보세요."
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
    r"큰\s*문제(?:는|가|도)?\s*없",           # "전반적으로 큰 문제는 없어 보입니다" (실측)
    r"문제(?:는|가|도)?\s*없어\s*보",
    r"안정적인\s*상태",                      # "전반적으로 안정적인 상태를 유지" (실측)
    r"전반적으로\s*안정",
)]

# --- G10: 예시 오염 ---
# 프롬프트 작성 예시(prompts/*.md)에 쓴 관찰 문구. 예시는 p2 프로파일의 보호자 의견을
# 재료로 썼는데, 7.8B 모델이 다른 프로파일에서 이 문구를 그대로 질문에 옮겨 적는
# 결함이 실측됐다 (보호자가 하지 않은 말을 인용). 입력 caregiver_notes에 같은 문구가
# 있으면 정당한 인용이므로 위반이 아니다.
# "딴 데를" 같은 짧은 구는 보호자 문장의 정당한 바꿔 쓰기("딴 곳을")와 구분이 안 돼 제외한다.
EXAMPLE_PHRASES = ("학원 숙제", "놀이터", "또래에게 먼저 말")


# --- G6: 처방·치료 권고 ---
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

# --- G7: 형식 누출 (보호자 노출 텍스트에 코드·식별자가 새는 경우) ---
# 한국어 본문에 영문 소문자 식별자가 나타날 정당한 이유는 없다. 대문자 약어
# (T점수, K-CBCL, SEM, TRF)는 대상이 아니다. 한글 바로 옆에 붙은 경우도 잡도록
# \b 대신 영숫자 lookaround를 쓴다.
_SCALE_ID_ALT = "|".join(sorted(SCALE_NAMES, key=len, reverse=True))
FORMAT_LEAK_PATTERNS = [re.compile(p) for p in (
    r"scale_id",
    rf"(?<![A-Za-z0-9_])(?:{_SCALE_ID_ALT})(?![A-Za-z0-9_])",
    r"(?<![A-Za-z0-9_])[a-z][a-z0-9_]{2,}(?![A-Za-z0-9_])",
    r"\([^()]*[a-z_]{2,}[^()]*\)",
)]

# --- G8: 밴드 라벨 정합 ---
# 허용 어휘는 보고서 라벨 그대로(정상/준임상/임상)뿐이다. "임상적", "임상 판단",
# "비정상", "정상적"처럼 라벨이 아닌 일반어 용법은 매칭에서 제외한다.
BAND_WORD_PATTERN = re.compile(
    r"준임상|(?<!준)임상(?!적|가|\s*(?:심리|판단|해석|전문))|(?<!비)정상(?!적|화)")
BAND_OF_WORD = {"정상": "normal", "준임상": "borderline", "임상": "clinical"}
NONSTANDARD_BAND_PATTERNS = [re.compile(p) for p in (
    r"경계\s*(?:수준|선|성|범위|구간|영역|점수|단계|상태)",
    r"경계(?:에|로)\s*(?:해당|위치|속|있|가깝|걸)",
    r"준\s*임계",                       # "준임상"의 오기 (실측에서 관찰)
    r"임계\s*(?:범위|수준|구간)",
    r"위험군",
    r"높은\s*편",
    r"낮은\s*편",
    r"(?<![A-Za-z])(?:borderline|clinical|normal)(?![A-Za-z])",
)]
# 본문에 언급된 척도를 scale_id로 매핑하는 한국어 척도명 사전 (표기 변형 포함)
SCALE_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "total_problems": ("총 문제행동", "총문제행동", "총 문제 행동"),
    "internalizing": ("내재화",),
    "externalizing": ("외현화",),
    "withdrawn": ("위축",),
    "somatic": ("신체증상", "신체 증상"),
    "anxious_depressed": ("우울/불안", "우울·불안", "우울-불안", "우울불안",
                          "우울 및 불안", "불안/우울"),
    "social_immaturity": ("사회적 미성숙", "사회적미성숙", "사회 미성숙"),
    "thought_problems": ("사고의 문제", "사고 문제", "사고문제"),
    "attention": ("주의집중", "주의 집중"),
    "delinquent": ("비행",),
    "aggressive": ("공격성",),
}
_SENTENCE_END = re.compile(r"[.!?\n]")


@dataclass
class Violation:
    rule_id: str          # "G1".."G9"
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

TASK_BLOCKS = {
    "explain": ("overview", "before_counseling"),
    "prep": ("questions_for_counselor", "observation_points", "counselor_briefing"),
}


def expected_blocks(profile: CBCLProfile, task: str) -> list[str]:
    """이 task에서 반드시 채워져야 하는 블록 id 목록."""
    return list(TASK_BLOCKS[task])


def split_blocks(profile: CBCLProfile, task: str, raw: dict) -> dict[str, object]:
    """출력 JSON을 블록 단위로 나눈다 (상위 스키마 통과 후에 호출)."""
    return {k: raw.get(k) for k in expected_blocks(profile, task)}


# ---------------------------------------------------------------- 검사기

def _check_example_contamination(block: str, text: str, profile: CBCLProfile) -> list[Violation]:
    """G10: 프롬프트 예시의 관찰 문구가 입력 보호자 의견에 없는데 본문에 등장."""
    notes = " ".join(profile.caregiver_notes)
    for phrase in EXAMPLE_PHRASES:
        if phrase in text and phrase not in notes:
            return [Violation("G10", block,
                              f"보호자 의견에 없는 예시 문구 인용: {phrase!r} (caregiver_notes에만 있는 관찰을 인용)")]
    return []


def _check_format_leak(block: str, text: str) -> list[Violation]:
    """G7: 보호자 노출 텍스트의 코드 누출. 첫 매칭 1건만 보고한다 (피드백 간결성)."""
    for pat in FORMAT_LEAK_PATTERNS:
        m = pat.search(text)
        if m:
            return [Violation("G7", block, f"보호자용 본문에 영문 코드 노출: {m.group(0)!r}")]
    return []


def scale_mentions(text: str) -> list[tuple[int, str]]:
    """본문에 언급된 척도를 (위치, scale_id)로 나열한다 (한국어 척도명 사전 기준)."""
    found: list[tuple[int, str]] = []
    for sid, aliases in SCALE_NAME_ALIASES.items():
        for alias in aliases:
            start = text.find(alias)
            while start != -1:
                found.append((start, sid))
                start = text.find(alias, start + 1)
    return sorted(found)


def _check_band_labels(block: str, text: str, profile: CBCLProfile,
                       own_scale: str | None = None,
                       own_dominates: bool = False) -> list[Violation]:
    """G8: 밴드 어휘 정합.

    비표준 밴드 표현은 그 자체로 위반. 표준 어휘(정상/준임상/임상)는 직전
    밴드 어휘 이후 같은 구간에 언급된 척도(없으면 같은 문장 뒤쪽의 척도,
    그래도 없으면 블록의 own_scale)의 실제 band와 대조한다.
    own_dominates는 척도 해설 블록용: 구간에 자기 척도가 언급되면 자기
    척도만 대조한다 (종합지표 해설이 하위 척도를 나열하는 문장 보호).
    귀속할 척도를 못 찾으면 대조하지 않는다 (규칙 한계, README 명시).
    """
    found: list[Violation] = []
    for pat in NONSTANDARD_BAND_PATTERNS:
        m = pat.search(text)
        if m:
            found.append(Violation(
                "G8", block, f"허용되지 않는 밴드 표현 {m.group(0)!r} (정상/준임상/임상만 사용)"))
            break
    mentions = scale_mentions(text)
    scale_map = profile.scale_map()
    prev_end = 0
    for m in BAND_WORD_PATTERN.finditer(text):
        word, claimed = m.group(0), BAND_OF_WORD[m.group(0)]
        targets = [sid for pos, sid in mentions if prev_end <= pos < m.start()]
        if not targets:
            tail = _SENTENCE_END.search(text, m.end())
            sent_end = tail.start() if tail else len(text)
            nxt = BAND_WORD_PATTERN.search(text, m.end())
            if nxt:
                sent_end = min(sent_end, nxt.start())
            targets = [sid for pos, sid in mentions if m.end() <= pos < sent_end]
        if own_scale and (not targets or (own_dominates and own_scale in targets)):
            targets = [own_scale]
        for sid in dict.fromkeys(targets):
            actual = scale_map[sid].band
            if actual != claimed:
                found.append(Violation(
                    "G8", block,
                    f"{SCALE_NAMES[sid]} 라벨 불일치: 본문 {word!r} != 입력 band {BAND_KO[actual]!r}"))
        prev_end = m.end()
    return found


def _check_text(block: str, text: str, profile: CBCLProfile,
                own_scale: str | None = None, own_dominates: bool = False,
                caregiver_facing: bool = True) -> list[Violation]:
    """텍스트 1개에 대한 G1/G2/G3/G6/G7/G8 검사.

    own_scale은 이 텍스트가 속한 척도(scale 블록의 scale_id, 항목의
    source_scale). caregiver_facing=False(상담사용 사전 요약)는 G7을 건너뛴다.
    """
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
    if caregiver_facing:
        found += _check_format_leak(block, text)
    found += _check_band_labels(block, text, profile, own_scale, own_dominates)
    found += _check_example_contamination(block, text, profile)
    return found


def _require_str(block: str, value, name: str) -> list[Violation]:
    if not isinstance(value, str) or not value.strip():
        return [Violation("G5", block, f"{name}: 문자열 필수")]
    return []


def check_top_schema(task: str, raw) -> list[Violation]:
    """출력 최상위 구조 검사 (여기서 걸리면 전체 재생성)."""
    if not isinstance(raw, dict):
        return [Violation("G5", "*", "출력이 JSON 객체가 아님")]
    missing = [k for k in TASK_BLOCKS[task] if k not in raw]
    if missing:
        return [Violation("G5", "*", f"필수 키 누락: {missing}")]
    return []


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
    scale_map = profile.scale_map()
    # G9는 앵커가 될 비정상 척도가 있을 때만 적용한다 (전 척도 정상 프로파일 예외)
    apply_g9 = bool(profile.elevated_scales())
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            found.append(Violation("G5", f"{block}[{i}]", "항목은 객체여야 함"))
            continue
        sid = it.get("source_scale")
        scale = scale_map.get(sid)
        vs = _require_str(f"{block}[{i}]", it.get(text_key), text_key)
        found += vs
        if not vs:
            found += [Violation(v.rule_id, block, v.matched)
                      for v in _check_text(block, it[text_key], profile,
                                           own_scale=sid if scale else None)]
        if scale is None:
            found.append(Violation("G4", block, f"source_scale 매칭 실패: {sid!r}"))
        elif apply_g9 and scale.band == "normal":
            found.append(Violation(
                "G9", block,
                f"source_scale {sid}({SCALE_NAMES[sid]})는 정상 범위 - 질문·관찰의 근거는 준임상/임상 척도만"))
    return found


def check_block(profile: CBCLProfile, task: str, block: str, content) -> list[Violation]:
    """블록 1개에 대한 전체 규칙 검사."""
    if isinstance(content, dict) and content.get("_fallback"):
        return []  # 사전 작성 고정 문구
    if block == "questions_for_counselor":
        return _check_items_block(profile, block, content, "question", 5, 7)
    if block == "observation_points":
        return _check_items_block(profile, block, content, "point", 3, 5)
    # overview / limits / before_counseling / counselor_briefing: 순수 텍스트
    vs = _require_str(block, content, block)
    if vs:
        return vs
    # 상담사용 사전 요약은 보호자에게 노출되지 않으므로 G7(형식 누출) 대상이 아니다
    return _check_text(block, content, profile,
                       caregiver_facing=(block != "counselor_briefing"))


def check_output(profile: CBCLProfile, task: str, raw) -> list[Violation]:
    """출력 전체 일괄 검사 (하네스의 seeded 검사와 최종 잔존 위반 스캔용)."""
    top = check_top_schema(task, raw)
    if top:
        return top
    found: list[Violation] = []
    for block, content in split_blocks(profile, task, raw).items():
        found += check_block(profile, task, block, content)
    return found


# ---------------------------------------------------------------- 폴백

def fallback_for(profile: CBCLProfile, task: str, block: str):
    """검증에 끝내 실패한 블록을 대체할 사전 작성 안전 문구 (fail-closed)."""
    if block == "overview":
        return SAFE_OVERVIEW
    if block == "questions_for_counselor":
        return [{"question": SAFE_QUESTION, "source_scale": "total_problems", "_fallback": True}]
    if block == "observation_points":
        return [{"point": SAFE_OBSERVATION, "source_scale": "total_problems", "_fallback": True}]
    if block == "counselor_briefing":
        return SAFE_BRIEFING
    return SAFE_GENERIC_TEXT  # before_counseling


def rebuild(profile: CBCLProfile, task: str, blocks: dict[str, object]) -> dict:
    """블록 딕셔너리를 출력 스키마 형태로 재조립한다."""
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

    근거 필드(source_scale)를 가진 것은 prep의 질문·관찰 포인트뿐이다
    (explain은 0/0). 폴백으로 대체된 항목은 분모에서 제외한다.
    """
    valid_ids = set(profile.scale_map())
    have, need = 0, 0
    if task != "prep":
        return have, need
    for key in ("questions_for_counselor", "observation_points"):
        for item in output.get(key, []):
            if isinstance(item, dict) and item.get("_fallback"):
                continue
            need += 1
            if isinstance(item, dict) and item.get("source_scale") in valid_ids:
                have += 1
    return have, need
