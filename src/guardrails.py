"""출력 안전성 가드레일 (규칙 엔진, LLM 미사용).

LLM이 쓰는 블록은 1개뿐이다 (ADR 0010과 그 보강): prep 태스크의 질문(questions_for_counselor).
연결 문단, 가정 관찰 포인트, 상담사 요약은 결정론 조립(report_html.build_overview_text,
build_observation_points, build_counselor_briefing), 척도 카드 해설과 한계 고지(scale_texts.py),
상담 전 안내(report_html.PRE_COUNSELING_NOTE, ADR 0009)는 고정 문구라 검사 대상이 아니다.
아래 규칙은 전부 질문 항목의 문장에 적용된다.

출력 규칙 12종:
  G1 진단명 사전         - 진단명 등장 자체를 위반 처리 (부정문 포함, 의도된 과검출)
  G2 심각성 단정(양방향) - 심각 쪽 단정과 근거 없는 낙관 보증을 모두 차단
  G3 수치 금지           - 문장에 아라비아 숫자, 그리고 한글 수사 점수(점/T/퍼센트 앞 "일흔다섯 점",
                           점수 어휘 뒤 "T점수가 육십칠", 단독 십 단위 "예순일곱")가 있으면 위반.
                           수치는 화면 카드가 보여주므로 질문에 쓸 이유가 없다
  G4 근거 링크           - 항목의 source_scale 이 입력의 실제 척도와 매칭되는지
  G5 스키마와 문형       - 출력 JSON 구조, 필수 필드, 항목 수(질문 5~7). 질문은 1문장 의문형
                           (요?/까?) 25~90자. 문장 수는 따옴표 밖에서만 센다 (마침표까지 원문 그대로
                           인용한 「」는 1문장)
  G6 처방·치료 권고      - 약물·약 복용, 치료 받기·시작·고려(조사 무관), 병원·진료·상담센터 방문
                           지시, 정신건강의학과·소아청소년과·전문의 의뢰 시사 차단
                           (허용 형태는 "예약된 상담에서 상담사와 이야기해 보세요" 하나뿐)
  G7 형식 누출           - 문장에 scale_id 값, 영문 소문자 식별자, "scale_id", 괄호 안 영문 코드
  G8 밴드 라벨 정합      - 밴드 어휘는 보고서 라벨(정상/준임상/임상)만 허용하고, 언급된
                           척도(척도명 사전으로 매핑)의 실제 band와 일치해야 함.
                           "경계 수준", "경계성", "borderline", "위험군", "높은 편" 등은 위반
  G9 정상 척도 근거 금지 - source_scale band가 normal이면 위반 (종합지표 포함). 전 척도 정상
                           프로파일에서는 total_problems만 허용한다 (프롬프트 계약과 동일)
  G10 근거 강제          - 항목마다 (a) 어느 보호자 의견의 연속 6자 이상 조각(공백 제외)을 그대로
                           포함하거나 (b) source_scale 이 준임상 이상 척도(전 척도 정상이면
                           total_problems)여야 통과, 둘 다 아니면 위반. 문장에 척도명이 나오면 그
                           척도가 source_scale 과 같아야 한다 ("위축", "비행"은 척도 어휘나 조사가
                           붙을 때만 척도명). 인용 주장(보고 동사의 존대 회상형 "~다고 하셨는데",
                           "적어 주신", "보셨다고", "관찰하셨듯이"와 따옴표 인용)은 (a)를 만족해야만
                           허용. "~이라고 하는 범위"류 용어 풀이와 용어를 감싼 따옴표(“준임상”)는
                           인용 주장이 아니다. 프롬프트 작성 예시의 관찰 문구("학원 숙제", "놀이터"
                           등)가 이 프로파일의 의견에 없는데 등장하면 위반
  G11 질문 방향          - 질문은 보호자가 상담사에게 묻는 문형만. 보호자에게 되묻는 명백한 문형
                           ("알려주시겠어요", "말씀해 주세요", "사례를 더", "있으신가요")은 차단
  G12 위기 어휘 출력 금지 - LLM이 만든 보호자 노출 문장에 입력 게이트와 같은 CRISIS_PATTERNS가
                           걸리면 위반. 입력에 없던 위기 표현("없어지고 싶다고 말한 날")을 모델이
                           지어내거나 바꿔 쓴 경우를 막는다. 입력에 있었다면 게이트가 먼저 막았다

위반 블록은 최대 2회 재생성하고, 그래도 실패하면 사전 작성 안전 문구로
대체한다 (fail-closed). 리포트가 아예 안 나가는 일은 없고, 검증 안 된
문장이 나가는 일도 없다.

G5는 필수 키 누락만 잡는다. 스키마 밖 키(옛 스키마의 overview, counselor_briefing,
observation_points 등)는 위반이 아니라 무시되며, split_blocks/rebuild가 버리므로 리포트에 닿지 않는다.

입력 게이트 1종: 위기 신호 검출 (detect_crisis_signals). 보호자 의견에
긴급 키워드가 있으면 LLM 호출 자체를 하지 않는다 (generator가 이 함수로
차단하고, 화면은 상담 연결 안내만 출력한다). 사전은 보수적으로 과검출을
허용하되, 아동 자신의 공격 행동("동생을 때려요")은 공격성 척도의 정당한 관찰이라
위기가 아니다: 폭력은 가해 주체(어른, 형제, 또래)나 피해 표지("~한테 맞아요")가
있을 때만 잡는다. 같은 사전을 G12가 출력에도 적용한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parser import BAND_KO, SCALE_NAMES, CBCLProfile

MAX_REGEN = 2  # 첫 생성 이후 블록 단위 재생성 횟수

SAFE_GENERIC_TEXT = "이 부분의 자동 생성 문구는 검증을 통과하지 못했습니다. 예약된 상담에서 상담사에게 직접 들으시길 권합니다."
SAFE_QUESTION = "이번 결과에서 무엇부터 살펴보면 좋을지, 예약된 상담에서 상담사에게 직접 여쭤보시길 권합니다."

# 보호자 의견 안의 아동 이름을 가리는 대체어. LLM에는 마스킹된 의견이 들어가므로 (generator.profile_payload)
# G10의 인용 대조도 원문과 마스킹본 양쪽을 본다.
MASK_TOKEN = "아이"

# --- G1: 진단명 사전 (부정문·완곡형 포함, 등장 자체를 위반 처리) ---
DIAGNOSIS_PATTERN = re.compile(
    r"(ADHD|주의력\s*결핍|자폐|우울증|불안\s*장애|틱\s*장애|품행\s*장애"
    r"|발달\s*장애|학습\s*장애|조현병|강박\s*장애|반항\s*장애|공황\s*장애"
    r"|양극성|조울|우울\s*장애|적응\s*장애)",
    re.IGNORECASE,
)

# --- G2: 심각성 단정 (양방향) ---
SEVERITY_PATTERNS = [re.compile(p) for p in (
    r"심각",
    r"위중",
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
# 프롬프트 작성 예시(prompts/counsel_prep_system.md)에 쓴 관찰 문구. 예시는 p2 프로파일의 보호자
# 의견을 재료로 썼는데, 초기 실험용 소형 로컬 모델이 다른 프로파일에서 이 문구를 그대로 질문에
# 옮겨 적는 결함이 실측됐다 (보호자가 하지 않은 말을 인용). 입력 caregiver_notes에 같은 문구가
# 있으면 정당한 인용이므로 위반이 아니다.
# "딴 데를" 같은 짧은 구는 보호자 문장의 정당한 바꿔 쓰기("딴 곳을")와 구분이 안 돼 제외한다.
EXAMPLE_PHRASES = ("학원 숙제", "놀이터", "또래에게 먼저 말")
# G10: 인용 주장. 이 표현이 있으면 문장 안에 보호자 의견의 원문 조각이 실제로 있어야 한다.
# 보고 동사에 존대 회상형(셨/신/셔서)이 붙은 꼴만 인용 주장으로 본다. "~이라고 하는 범위",
# "~라고 합니다"는 용어 풀이 문형(quality.GLOSS_PATTERNS)이라 인용 주장이 아니다.
_REPORT_VERBS = (r"(?:적어|적으|써|쓰|말씀해|말씀하|말해|말하|말씀|하|얘기하|이야기하|전해|전하"
                 r"|느끼|보|표현하|언급하|기록하|보고하|남겨|남기)")
QUOTE_CLAIM_PATTERNS = [re.compile(p) for p in (
    r"(?:적어|적으|써|쓰|남겨|남기)\s*(?:주)?(?:셨|신)",                     # 적어 주셨는데, 적으신
    rf"(?:다|라|냐|자)고\s*{_REPORT_VERBS}\s*(?:주)?(?:셨|신|셔서)",         # ~다고 하셨는데, ~라고 적으셨는데
    r"(?:보|하|느끼|말씀하|말하|적|쓰|기록하|관찰하|경험하|겪으|들으)(?:셨|신)(?:다|대)고",  # ~보셨다고
    r"(?:관찰|말씀|언급|보고|기록|경험|표현|묘사|서술)하(?:셨|신)",            # 관찰하셨듯이
)]
# 따옴표 쌍. 안의 문자열이 척도명·밴드 라벨·용어(아래 _TERM_QUOTE)면 인용이 아니라 용어 표시다.
QUOTE_PAIRS = (("「", "」"), ("“", "”"), ("‘", "’"), ("『", "』"), ("〈", "〉"), ('"', '"'), ("'", "'"))
QUOTE_MIN_CHARS = 6  # (a) 인용으로 인정하는 연속 글자 수 (공백 제외)


# --- G6: 처방·치료 권고 ---
PRESCRIPTION_PATTERNS = [re.compile(p) for p in (
    r"약물",
    r"약을\s*(?:복용|먹)",
    r"약(?:을|물|\s)*(?:복용|처방)",                                       # 약 복용, 약물 복용
    r"치료(?:를|가|는|도|만|까지|부터|라도)?\s*(?:받|필요|시작|고려|권|알아보)",  # 조사가 바뀌어도 잡는다
    r"치료\s*프로그램",
    r"(?:놀이|미술|음악|언어|인지\s*행동|심리|행동)\s*치료",
    r"병원(?:에|을|으로|에도|부터|이라도)?\s*(?:가|방문|데려|예약|들르|찾)",   # 병원에 데려가기, 병원 예약
    r"진료(?:를|도|는)?\s*(?:받|예약|보)",
    r"처방",
    r"의료\s*기관",
    r"전문\s*기관",
    r"상담\s*센터(?:에|를|도)?\s*(?:등록|방문|찾|가)",
    # 완곡한 의뢰 시사도 차단: "전문의 상담이 필요한 수준으로 보입니다" 류.
    # 허용된 전문가 안내 형태는 "예약된 상담에서 상담사와..." 하나뿐이다.
    r"전문의",
    r"정신과",
    r"정신\s*건강\s*의학과",
    r"소아\s*정신",
    r"소아\s*청소년\s*(?:정신)?과",
)]

# --- G3: 수치 금지 ---
# 아라비아 숫자는 자리와 무관하게 위반이다. 한글 수사는 세 자리에서 본다.
#   (1) 점/T/퍼센트 앞 ("일흔다섯 점", "육십 칠 점" - 조각 사이 공백 허용)
#   (2) 점수 어휘 뒤 ("T점수가 육십칠", "백분위 구십", "척도가 예순일곱")
#   (3) 단독 순우리말 십 단위 ("예순일곱이면", "일흔다섯") - 서른 이상은 수사 외의 용법이 없다
# "이 점은", "매일 점심", "두 번"처럼 수사가 아닌 흔한 음절과 겹치는 단독 음절은 제외한다.
# "열"(열어, 열심히)과 "쉰"(쉰 뒤에), "스무"는 단독으로 세지 않고 (1) 또는 단위 수사가 붙을 때만 본다.
ARABIC_DIGIT_PATTERN = re.compile(r"\d")
_SINO_TENS = r"(?:[일이삼사오육륙칠팔구]\s*)?십(?:\s*[일이삼사오육륙칠팔구])?"   # 육십칠, 칠십, 육십 칠
_NATIVE_UNITS = r"(?:하나|한|둘|두|셋|세|넷|네|다섯|여섯|일곱|여덟|아홉)"
_NATIVE_TENS_SAFE = r"(?:스물|서른|마흔|예순|일흔|여든|아흔)"
_NATIVE_TENS_ANY = r"(?:열|스물|스무|서른|마흔|쉰|예순|일흔|여든|아흔)"
_NUMERAL_BEFORE_SCORE = (rf"(?:{_SINO_TENS}|백|{_NATIVE_TENS_ANY}(?:\s*{_NATIVE_UNITS})?"
                         r"|(?:다섯|여섯|일곱|여덟|아홉))")
_NUMERAL_AFTER_SCORE = (rf"(?:{_SINO_TENS}|{_NATIVE_TENS_SAFE}(?:\s*{_NATIVE_UNITS})?"
                        rf"|(?:열|쉰|스무)\s*{_NATIVE_UNITS})")
_SCORE_WORD_BEFORE = r"(?:T\s*점수|백분위|점수|척도|결과|수치)(?:가|는|이|를|은|로|도|의|에서)?\s*"
KOREAN_NUMERAL_SCORE_PATTERNS = [re.compile(p) for p in (
    rf"{_NUMERAL_BEFORE_SCORE}\s*(?:점(?!검)|T(?![A-Za-z])|퍼센트|%)",
    rf"{_SCORE_WORD_BEFORE}{_NUMERAL_AFTER_SCORE}",
    rf"{_NATIVE_TENS_SAFE}(?:\s*{_NATIVE_UNITS})?|쉰\s*{_NATIVE_UNITS}",
)]

# --- G7: 형식 누출 (문장에 코드·식별자가 새는 경우) ---
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
    r"경계\s*(?:수준|선|성|범위|구간|영역|점수|단계|상태|군)",
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
# 일상어와 겹치는 별칭("위축된", "비행기")은 척도 어휘가 뒤따르거나 조사가 바로 붙을 때만 척도 언급으로 본다.
_ALIAS_CONTEXT = re.compile(
    r"(?=\s*(?:척도|문제|영역|점수|결과|범위|구간|수준|항목)"
    r"|(?:이|가|은|는|을|를|과|와|의|도|에|로|만|이나|이라|라는|처럼|까지|부터|에서|만큼|보다|이며|이고|하고|이란)"
    r"|\s*[,、·/])")
_CONTEXT_ALIASES = {"위축", "비행"}
_SENTENCE_END = re.compile(r"[.!?\n]")
# 따옴표 안 문자열이 이 꼴이면 인용이 아니라 용어 표시다 (“준임상”이라는 말은 ...). 공백 제거 후 대조.
_TERM_QUOTE = re.compile(
    r"^(?:정상|준임상|임상)(?:범위|구간|수준)?$"
    r"|^(?:" + "|".join(sorted({re.escape(re.sub(r"\s+", "", a)) for al in SCALE_NAME_ALIASES.values() for a in al},
                             key=len, reverse=True)) + r")(?:척도|문제|영역|점수)?$"
    r"|^(?:T점수|백분위|퍼센트|표준편차|규준|증후군|SEM|신뢰구간|오차범위)$")

# --- G5: 문형 ---
QUESTION_MIN_CHARS, QUESTION_MAX_CHARS = 25, 90
QUESTION_END_PATTERN = re.compile(r"(?:요|까|죠)\s*\?$")     # 의문형 종결: ~나요? ~까요? ~습니까? ~죠?
_TERMINATORS = re.compile(r"[.?!]")

# --- G11: 질문 방향 ---
# 상담사가 보호자에게 되묻는 명백한 문형. 양방향 모두에 쓰이는 표현("설명해 주실 수 있나요")은
# 여기 넣지 않고 quality.REVERSE_DIRECTION_PATTERNS(WARN)에 남긴다.
REVERSE_DIRECTION_BLOCK_PATTERNS = [re.compile(p) for p in (
    r"알려\s*주시겠",
    r"알려\s*주시면",
    r"말씀해\s*주(?:시|세요)",
    r"공유해\s*주(?:시|세요)",
    r"사례를\s*(?:더|몇)",
    r"(?:관찰|언급|경험|말씀)하신",
    r"있으신가요",
    r"보호자님|어머님|아버님|부모님께서",
)]


@dataclass
class Violation:
    rule_id: str          # "G1".."G12"
    block: str            # 위반이 발견된 블록 id (예: "questions_for_counselor")
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
# 오검출 여러 건의 비용보다 크다. 어간 뒤의 활용("때려요", "맞아요", "그었어요", "만져요",
# "죽는 게 낫겠다")까지 잡도록 어간 + 활용 어미 꼴로 쓴다.
# 폭력 어휘의 경계: 아동이 남을 때리는 관찰("동생을 때려요")은 공격성 척도의 정당한 입력이라
# 위기로 보지 않는다. 가해 주체(_ABUSER, "아빠가 때려요"), 피해 대상(_VICTIM, "아이를 때려요"),
# 피해 표지("~한테 맞아요", "맞고 왔어요", "맞은 자국")가 있을 때만 잡는다.
# 동의의 "맞아요", 관용구 "죽도록", "때려치우다", 비의 "비 맞고"는 걸리지 않아야 한다 (테스트로 고정).
_ADV = (r"(?:자주|매일|가끔|늘|계속|심하게|많이|막|자꾸|툭하면|밤마다|종종|세게|손으로|주먹으로|몽둥이로"
        r"|화나면|화가\s*나면|술\s*마시면|술만\s*마시면|술을\s*마시면|말을\s*안\s*들으면|시험을\s*못\s*보면)")
_ABUSER = (r"(?:아빠|아버지|엄마|어머니|부모님?|새아빠|새엄마|계부|계모|남편|아내|삼촌|외삼촌|이모부?|고모부?"
           r"|큰아빠|작은아빠|큰엄마|작은엄마|할머니|할아버지|형|오빠|누나|언니|선생님|학원\s*선생님?|코치|원장"
           r"|어른|누군가|누가|아저씨|친구들|애들|형들|동네\s*형)")
_VICTIM = r"(?:저|나|아이|애|얘|자기|딸|아들|우리\s*애|우리\s*아이|자신)"
_BODY = r"(?:손목|팔목|팔뚝|팔|허벅지|다리|배|몸|살|피부|목)"
_BLADE = r"(?:칼|커터|커터칼|면도날|유리\s*조각|날카로운\s*것)"
_PRIVATE = r"(?:몸|가슴|성기|고추|음부|엉덩이|허벅지|속옷\s*안|중요한\s*부위|은밀한\s*부위)"
CRISIS_PATTERNS = [re.compile(p) for p in (
    r"자해",
    r"자살",
    r"차라리\s*죽겠",
    r"죽고\s*싶",
    r"죽고싶",
    r"죽어\s*버(?:리|릴|렸)",
    r"죽었으면",
    r"죽는\s*(?:게|것이|편이)\s*(?:더\s*)?(?:낫|나을)",
    r"죽어야\s*겠",
    r"죽여\s*달라",
    r"극단적\s*선택",
    r"(?:사는|살아갈)\s*이유가\s*없",
    r"살고\s*싶지\s*않",
    r"살기\s*싫",
    r"사라(?:지고|져\s*버리고)\s*싶",
    r"사라져\s*버리",
    r"없어(?:지고|져\s*버리고)\s*싶",
    rf"{_BODY}(?:을|를|에|에다)?\s*(?:{_BLADE}(?:로|으로)\s*)?(?:긋|그(?:었|어|으|은|을))",
    rf"{_BLADE}(?:로|으로)?\s*(?:[가-힣]+(?:을|를)\s*)?(?:긋|그(?:었|어|으|은|을))",
    r"뛰어\s*내리(?:고\s*싶|겠|려)",
    r"(?:옥상|창문|다리|베란다).{0,15}뛰어\s*내(?:렸|린)",
    r"목(?:을\s*)?(?:매(?:\s*(?:겠|려고|려|었|고\s*싶)|\s*$)|맸)",
    r"(?:약|수면제|진통제|알약)(?:을|를)?\s*(?:한꺼번에|몽땅|전부)(?:\s*(?:먹|삼키|복용))?",
    r"목숨",
    r"스스로\s*(?:(?:자신|몸)(?:을|를)\s*)?(?:를\s*)?(?:해치|다치|상처)",
    r"몸에\s*상처",
    r"상처를\s*(?:내|냈|낸)",
    r"맞(?:아서|고)\s*(?:멍|상처|피)",
    r"맞은\s*(?:자국|흔적|멍)",
    rf"(?:한테|에게|께|한테서|에게서)\s*(?:{_ADV}\s*)*(?:[가-힣]+(?:을|를)\s*)?맞(?:아|았|는|고|습|을|기|네|은)",
    r"(?<![비눈물공볼을를])(?<![비눈물공볼을를]\s)맞고\s*(?:다니|다녀|들어오|들어왔|왔)",
    r"(?:때린다고|때리겠다고|때린대|때렸다고)",
    rf"{_ABUSER}(?:이|가|께서|은|는|도|들이)?\s*(?:{_ADV}\s*)*때(?:려|린|렸|리|립|릴|림)(?!치)",
    rf"{_VICTIM}(?:를|을)\s*(?:{_ADV}\s*)*때(?:려|린|렸|리|립|릴|림)(?!치)",
    r"폭행(?:을)?\s*(?:당|받)",
    r"학대(?:를)?\s*(?:당|받)",
    r"성적(?:인)?\s*접촉",
    r"성추행",
    r"성폭력",
    rf"{_PRIVATE}(?:을|를|에|에다)?\s*(?:자꾸\s*|계속\s*|억지로\s*|몰래\s*|함부로\s*)?만(?:져|졌|지(?!작)|진)",
    r"옷을\s*벗기(?:려|려고|었|였|겠)",
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
    "prep": ("questions_for_counselor",),
}
ITEM_TEXT_KEY = {"questions_for_counselor": "question"}
ITEM_COUNT = {"questions_for_counselor": (5, 7)}


def expected_blocks(profile: CBCLProfile, task: str) -> list[str]:
    """이 task에서 반드시 채워져야 하는 블록 id 목록."""
    return list(TASK_BLOCKS[task])


def split_blocks(profile: CBCLProfile, task: str, raw: dict) -> dict[str, object]:
    """출력 JSON을 블록 단위로 나눈다 (상위 스키마 통과 후에 호출)."""
    return {k: raw.get(k) for k in expected_blocks(profile, task)}


# ---------------------------------------------------------------- 인용 대조 (G10)

def mask_notes(notes: list[str], alias: str) -> list[str]:
    """보호자 의견 안에 아동 이름이 적혀 있으면 MASK_TOKEN으로 바꾼다.

    LLM 입력(generator.profile_payload)과 G10의 인용 대조가 같은 함수를 쓴다. 이름 뒤 조사는
    그대로 둔다 ("민수가" 는 "아이가").
    """
    alias = (alias or "").strip()
    if not alias:
        return list(notes)
    return [n.replace(alias, MASK_TOKEN) for n in notes]


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def quotable_notes(profile: CBCLProfile) -> list[str]:
    """인용 대조 기준이 되는 보호자 의견: 원문과 마스킹본 (중복 제거)."""
    raw = [n for n in profile.caregiver_notes if n and n.strip()]
    masked = mask_notes(raw, profile.child.alias)
    return list(dict.fromkeys(raw + masked))


def quotes_caregiver_note(text: str, notes: list[str], min_chars: int = QUOTE_MIN_CHARS) -> bool:
    """text가 어느 보호자 의견의 연속 min_chars자 이상 조각(공백 제외)을 그대로 담고 있는지."""
    t = _squash(text)
    for note in notes:
        n = _squash(note)
        if not n:
            continue
        if len(n) <= min_chars:
            if n in t:
                return True
            continue
        for i in range(len(n) - min_chars + 1):
            if n[i:i + min_chars] in t:
                return True
    return False


def quoted_spans(text: str) -> list[str]:
    """짝이 맞는 따옴표(QUOTE_PAIRS) 안의 문자열 목록."""
    inner: list[str] = []
    for open_q, close_q in QUOTE_PAIRS:
        inner += re.findall(re.escape(open_q) + r"([^" + re.escape(close_q) + r"]*)" + re.escape(close_q), text)
    return inner


def strip_quoted(text: str) -> str:
    """짝이 맞는 따옴표 안을 비운다 (따옴표 기호는 남긴다). G5가 따옴표 밖 문장 수만 세는 데 쓴다."""
    for open_q, close_q in QUOTE_PAIRS:
        text = re.sub(re.escape(open_q) + r"[^" + re.escape(close_q) + r"]*" + re.escape(close_q),
                      open_q + close_q, text)
    return text


def has_quote_claim(text: str) -> bool:
    """G10: 문장이 보호자의 말을 인용한다고 주장하는지.

    (1) 보고 동사의 존대 회상형("~다고 하셨는데", "적어 주신", "보셨다고", "관찰하셨듯이"),
    (2) 따옴표 인용. 따옴표 안이 척도명·밴드 라벨·용어뿐이면(“준임상”이라는 말은) 용어 표시라 제외하고,
    짝이 안 맞는 따옴표는 보수적으로 인용으로 본다.
    """
    if any(p.search(text) for p in QUOTE_CLAIM_PATTERNS):
        return True
    if any(not _TERM_QUOTE.match(_squash(inner)) for inner in quoted_spans(text)):
        return True
    leftover = strip_quoted(text)
    for open_q, close_q in QUOTE_PAIRS:
        leftover = leftover.replace(open_q + close_q, "")
    return any(q in leftover for pair in QUOTE_PAIRS for q in pair)


# ---------------------------------------------------------------- 검사기

def _check_example_contamination(block: str, text: str, profile: CBCLProfile) -> list[Violation]:
    """G10: 프롬프트 예시의 관찰 문구가 입력 보호자 의견에 없는데 본문에 등장."""
    notes = " ".join(profile.caregiver_notes)
    for phrase in EXAMPLE_PHRASES:
        if phrase in text and phrase not in notes:
            return [Violation("G10", block,
                              f"보호자 의견에 없는 예시 문구 인용: {phrase!r} (caregiver_notes에만 있는 관찰을 인용)")]
    return []


def _check_grounding(block: str, text: str, profile: CBCLProfile, sid, scale) -> list[Violation]:
    """G10: 근거 강제. (a) 보호자 의견 인용 또는 (b) 준임상 이상 근거 척도, 척도명은 source_scale과 일치."""
    found: list[Violation] = []
    quoted = quotes_caregiver_note(text, quotable_notes(profile))
    all_normal = not profile.elevated_scales()
    anchored = scale is not None and (
        (all_normal and sid == "total_problems") or (not all_normal and scale.band != "normal"))
    if not quoted and not anchored:
        found.append(Violation(
            "G10", block,
            f"근거 없음: 보호자 의견의 원문 조각(연속 {QUOTE_MIN_CHARS}자)도 없고 source_scale {sid!r}도 준임상 이상 척도가 아님"))
    if not quoted and has_quote_claim(text):
        found.append(Violation(
            "G10", block, "인용 주장이 있으나 보호자 의견의 원문 조각이 없음 (보호자가 하지 않은 말을 인용)"))
    mentioned = [s for _pos, s in scale_mentions(text)]
    mismatched = [s for s in dict.fromkeys(mentioned) if s != sid]
    if mismatched:
        names = ", ".join(SCALE_NAMES[s] for s in mismatched)
        found.append(Violation(
            "G10", block,
            f"척도 불일치: 문장은 {names}을(를) 말하는데 source_scale은 {sid!r}"))
    found += _check_example_contamination(block, text, profile)
    return found


def _check_format_leak(block: str, text: str) -> list[Violation]:
    """G7: 문장의 코드 누출. 첫 매칭 1건만 보고한다 (피드백 간결성)."""
    for pat in FORMAT_LEAK_PATTERNS:
        m = pat.search(text)
        if m:
            return [Violation("G7", block, f"본문에 영문 코드 노출: {m.group(0)!r}")]
    return []


def scale_mentions(text: str) -> list[tuple[int, str]]:
    """본문에 언급된 척도를 (위치, scale_id)로 나열한다 (한국어 척도명 사전 기준).

    "위축", "비행"은 일상어("위축된 듯한", "비행기 소리")와 겹치므로 척도 어휘가 뒤따르거나
    조사가 바로 붙는 경우("위축 척도", "비행이 준임상")만 척도 언급으로 센다.
    """
    found: list[tuple[int, str]] = []
    for sid, aliases in SCALE_NAME_ALIASES.items():
        for alias in aliases:
            start = text.find(alias)
            while start != -1:
                if alias not in _CONTEXT_ALIASES or _ALIAS_CONTEXT.match(text, start + len(alias)):
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


def _check_numbers(block: str, text: str) -> list[Violation]:
    """G3: 아라비아 숫자 금지 + 한글 수사 점수 금지 (점/T/퍼센트 앞, 점수 어휘 뒤, 단독 십 단위)."""
    found: list[Violation] = []
    m = ARABIC_DIGIT_PATTERN.search(text)
    if m:
        run = re.search(r"\d[\d.,%]*", text)
        found.append(Violation("G3", block, f"아라비아 숫자: {run.group(0) if run else m.group(0)!r} (수치는 카드가 보여줌)"))
    for pat in KOREAN_NUMERAL_SCORE_PATTERNS:
        m = pat.search(text)
        if m:
            found.append(Violation("G3", block, f"한글 수사 점수: {m.group(0)!r}"))
            break
    return found


def _check_crisis_vocab(block: str, text: str) -> list[Violation]:
    """G12: 보호자 노출 문장에 위기 어휘. 입력 게이트(detect_crisis_signals)와 같은 사전을 쓴다.

    입력에 위기 표현이 있었다면 LLM은 호출되지 않았으므로, 여기서 걸리는 것은 모델이 지어내거나
    바꿔 쓴 표현이다. 첫 매칭 1건만 보고한다.
    """
    for pat in CRISIS_PATTERNS:
        m = pat.search(text)
        if m:
            return [Violation("G12", block, f"위기 어휘 출력: {m.group(0)!r} (입력에 없는 위기 표현을 보호자에게 노출)")]
    return []


def _check_text(block: str, text: str, profile: CBCLProfile,
                own_scale: str | None = None, own_dominates: bool = False) -> list[Violation]:
    """텍스트 1개에 대한 G1/G2/G3/G6/G7/G8/G12 검사.

    own_scale은 이 텍스트가 속한 척도(항목의 source_scale). G8의 밴드 귀속에 쓴다.
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
    found += _check_numbers(block, text)
    found += _check_format_leak(block, text)
    found += _check_band_labels(block, text, profile, own_scale, own_dominates)
    found += _check_crisis_vocab(block, text)
    return found


def _check_question_form(block: str, text: str) -> list[Violation]:
    """G5 문형: 질문은 1문장, 의문형 종결(요?/까?/죠?), 25~90자.

    문장 수는 따옴표 밖에서만 센다. 보호자 의견을 마침표까지 원문 그대로 「」로 인용한 질문은 1문장이다.
    """
    found: list[Violation] = []
    t = text.strip()
    n = len(t)
    if not (QUESTION_MIN_CHARS <= n <= QUESTION_MAX_CHARS):
        found.append(Violation("G5", block, f"질문 길이 {n}자 (요구: {QUESTION_MIN_CHARS}~{QUESTION_MAX_CHARS}자)"))
    if not QUESTION_END_PATTERN.search(t):
        found.append(Violation("G5", block, f"질문이 의문형(요?/까?)으로 끝나지 않음: {t[-12:]!r}"))
    outside = strip_quoted(t)
    if len(_TERMINATORS.findall(outside)) != 1 or "\n" in t:
        found.append(Violation("G5", block, "질문은 1문장이어야 함 (마침표·물음표가 둘 이상)"))
    return found


def _check_direction(block: str, text: str) -> list[Violation]:
    """G11: 보호자에게 되묻는 명백한 문형 차단 (질문 블록만)."""
    for pat in REVERSE_DIRECTION_BLOCK_PATTERNS:
        m = pat.search(text)
        if m:
            return [Violation("G11", block, f"보호자에게 되묻는 문형: {m.group(0)!r} (질문은 보호자가 상담사에게 묻는 방향만)")]
    return []


def _require_str(block: str, value, name: str) -> list[Violation]:
    if not isinstance(value, str) or not value.strip():
        return [Violation("G5", block, f"{name}: 문자열 필수")]
    return []


def check_top_schema(task: str, raw) -> list[Violation]:
    """출력 최상위 구조 검사 (여기서 걸리면 전체 재생성).

    필수 키 누락만 본다. 스키마 밖 키는 위반이 아니라 무시된다 (split_blocks가
    버리므로 리포트에 닿지 않는다). 옛 스키마의 overview, counselor_briefing도 마찬가지다.
    """
    if not isinstance(raw, dict):
        return [Violation("G5", "*", "출력이 JSON 객체가 아님")]
    missing = [k for k in TASK_BLOCKS[task] if k not in raw]
    if missing:
        return [Violation("G5", "*", f"필수 키 누락: {missing}")]
    return []


def _check_items_block(profile: CBCLProfile, block: str, items) -> list[Violation]:
    """항목 목록 블록(questions_for_counselor) 검사: 항목 수, 항목별 문장 규칙, 근거 척도."""
    text_key = ITEM_TEXT_KEY[block]
    lo, hi = ITEM_COUNT[block]
    if not isinstance(items, list):
        return [Violation("G5", block, "배열이어야 함")]
    real_items = [it for it in items if not (isinstance(it, dict) and it.get("_fallback"))]
    if real_items != items:  # 폴백 문구 블록은 재검사하지 않는다
        return []
    found: list[Violation] = []
    if not (lo <= len(items) <= hi):
        found.append(Violation("G5", block, f"항목 수 {len(items)}건 (요구: {lo}~{hi})"))
    scale_map = profile.scale_map()
    all_normal = not profile.elevated_scales()
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            found.append(Violation("G5", f"{block}[{i}]", "항목은 객체여야 함"))
            continue
        sid = it.get("source_scale")
        if sid is not None and not isinstance(sid, str):
            # 배열·객체 등 해시 불가 값은 스키마 위반이다. dict.get에 넘기면 TypeError로 죽어
            # "위반은 재생성, 상한 뒤 안전 문구"라는 fail-closed 계약을 지키지 못한다.
            found.append(Violation("G5", f"{block}[{i}]", f"source_scale은 문자열이어야 함: {sid!r}"))
            sid = None
        scale = scale_map.get(sid)
        vs = _require_str(f"{block}[{i}]", it.get(text_key), text_key)
        found += vs
        if not vs:
            text = it[text_key]
            found += [Violation(v.rule_id, block, v.matched)
                      for v in _check_text(block, text, profile, own_scale=sid if scale else None)]
            found += _check_question_form(block, text)
            found += _check_direction(block, text)
            found += _check_grounding(block, text, profile, sid, scale)
        if scale is None:
            found.append(Violation("G4", block, f"source_scale 매칭 실패: {sid!r}"))
        elif all_normal and sid != "total_problems":
            found.append(Violation(
                "G9", block,
                f"전 척도 정상 프로파일의 근거는 total_problems만 (source_scale {sid})"))
        elif not all_normal and scale.band == "normal":
            found.append(Violation(
                "G9", block,
                f"source_scale {sid}({SCALE_NAMES[sid]})는 정상 범위 - 질문의 근거는 준임상/임상 척도만"))
    return found


def check_block(profile: CBCLProfile, task: str, block: str, content) -> list[Violation]:
    """블록 1개에 대한 전체 규칙 검사."""
    if isinstance(content, dict) and content.get("_fallback"):
        return []  # 사전 작성 고정 문구
    if block in ITEM_TEXT_KEY:
        return _check_items_block(profile, block, content)
    return [Violation("G5", block, "알 수 없는 블록")]  # 방어용, 현행 스키마에서는 닿지 않음


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
    if block == "questions_for_counselor":
        return [{"question": SAFE_QUESTION, "source_scale": "total_problems", "_fallback": True}]
    return SAFE_GENERIC_TEXT  # 알 수 없는 블록 (방어용, 현행 스키마에서는 닿지 않음)


def rebuild(profile: CBCLProfile, task: str, blocks: dict[str, object]) -> dict:
    """블록 딕셔너리를 출력 스키마 형태로 재조립한다."""
    return {k: blocks[k] for k in expected_blocks(profile, task)}


INTERNAL_FLAGS = ("_fallback", "_pending")  # 이 저장소의 코드만 붙이는 내부 표식


def _strip_fallback_flags(value):
    """LLM 출력이 내부 표식을 흉내 내지 못하게 제거한다.

    _fallback은 이 모듈이 폴백을 삽입할 때만, _pending은 report_html.pending_results가 생성 전
    미리보기를 만들 때만 붙는다. LLM 출력에 섞여 오면 _fallback은 검사 우회, _pending은 화면의
    "생성 대기" 표식과 근거 배지 숨김을 위조하므로 둘 다 입구에서 벗긴다.
    """
    if isinstance(value, dict):
        return {k: _strip_fallback_flags(v) for k, v in value.items() if k not in INTERNAL_FLAGS}
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

    근거 필드(source_scale)를 가진 LLM 항목은 prep의 질문뿐이다 (관찰 포인트는 결정론 조립).
    폴백으로 대체된 항목은 분모에서 제외한다.
    """
    valid_ids = set(profile.scale_map())
    have, need = 0, 0
    if task != "prep":
        return have, need
    for key in ("questions_for_counselor",):
        for item in output.get(key, []):
            if isinstance(item, dict) and item.get("_fallback"):
                continue
            need += 1
            sid = item.get("source_scale") if isinstance(item, dict) else None
            if isinstance(sid, str) and sid in valid_ids:
                have += 1
    return have, need
