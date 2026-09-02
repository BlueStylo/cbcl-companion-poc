"""품질 지표 3종 (측정·표기 전용, 차단 아님).

가드레일(G1~G9)은 안전 규칙이라 위반이면 블록을 폐기하지만, 아래 지표는
"이 기획의 척추"가 지켜지는지를 재는 관측값이다. 하네스 요약 표의 별도
절과 main.py 실행 통계에 표기만 한다.

  (i)   전문 용어 잔존율   - 용어 사전 등장 횟수, 용어가 등장한 블록 중
                             같은 블록에 풀이 표현이 동반된 비율
  (ii)  보호자 표현 반영률 - caregiver_notes의 토큰이 질문·관찰 텍스트에
                             등장하는 비율 (항목 단위 / 토큰 단위)
  (iii) 질문 방향 경고     - 상담사가 보호자에게 되묻는 패턴을 WARN으로 집계.
                             의미 판정이라 규칙으로는 한계가 있어 차단하지 않는다.

측정 대상은 보호자에게 노출되는 생성 텍스트다. 상담사용 사전 요약은
전문 용어가 기대되는 문서라 (i)에서 제외하고, 폴백(사전 작성 안전 문구)은
생성 품질이 아니므로 모든 지표에서 제외한다.
"""

from __future__ import annotations

import re

from .parser import CBCLProfile

# ---------------------------------------------------------------- (i) 전문 용어

JARGON_TERMS: dict[str, re.Pattern] = {
    name: re.compile(p) for name, p in (
        ("T점수", r"T\s*점수"),
        ("백분위", r"백분위"),
        ("내재화", r"내재화"),
        ("외현화", r"외현화"),
        ("준임상", r"준임상"),
        ("임상", r"(?<!준)임상"),
        ("척도", r"척도"),
        ("증후군", r"증후군"),
        ("표준편차", r"표준\s*편차"),
        ("신뢰구간", r"신뢰\s*구간"),
        ("SEM", r"(?<![A-Za-z])SEM(?![A-Za-z])"),
        ("규준", r"규준"),
    )
}
GLOSS_PATTERNS = [re.compile(p) for p in (
    r"또래\s*100\s*명\s*중",
    r"평균(?:을|이|은|인)?\s*50",
    r"쉽게\s*말(?:하면|해)",
    r"(?:이)?라고\s*(?:부릅니다|불러요|합니다|해요)",
    r"(?:이|다)?라는\s*(?:뜻|의미)",
    r"다는\s*(?:뜻|의미)",
    r"다시\s*말해",
    r"풀어\s*(?:말하면|쓰면|보면)",
    r"(?:이란|란)\s",
)]

# ---------------------------------------------------------------- (ii) 표현 반영

# 조사·어미 제거 휴리스틱 (긴 것부터). 남는 어간이 2글자 미만이면 제거하지 않는다.
_SUFFIXES = sorted((
    "였습니다", "았습니다", "었습니다", "했습니다", "습니다", "입니다", "합니다",
    "어요", "아요", "해요", "에서는", "에서", "에게", "으로", "까지", "부터",
    "처럼", "보다", "이나", "마다", "라는", "이라는", "들이", "들을", "들은", "들의",
    "라고", "은", "는", "이", "가", "을", "를", "에", "의", "도", "와", "과", "로",
    "만", "면", "고", "서", "지", "다",
), key=len, reverse=True)
# 반영 여부를 재는 데 의미가 없는 흔한 말
_STOPWORDS = {
    "아이", "우리", "자주", "정말", "너무", "조금", "많이", "가끔", "요즘", "최근",
    "때문", "그리고", "하지만", "있어", "있습", "것이", "정도", "이런", "그런",
    "어떤", "무엇", "하나", "다시", "다른", "모두", "항상", "한번", "자꾸", "계속",
    "일이", "적이", "번", "않아", "싶습", "들어", "있는", "없는", "해서", "하고",
}
_HANGUL_TOKEN = re.compile(r"[가-힣]+")


def note_tokens(notes: list[str]) -> list[str]:
    """caregiver_notes에서 2글자 이상 명사·어간 후보를 뽑는다 (중복 제거, 순서 유지)."""
    out: list[str] = []
    for note in notes:
        for raw in _HANGUL_TOKEN.findall(note):
            tok = raw
            for suf in _SUFFIXES:
                if tok.endswith(suf) and len(tok) - len(suf) >= 2:
                    tok = tok[: -len(suf)]
                    break
            if len(tok) >= 2 and tok not in _STOPWORDS and tok not in out:
                out.append(tok)
    return out


# ---------------------------------------------------------------- (iii) 질문 방향

# 상담사가 보호자에게 되묻는 형태의 단서. 보호자→상담사 문장에도 쓰일 수 있는
# 표현이 있어(예: "설명해 주실 수 있나요") 의미 판정은 규칙의 한계 - WARN만.
REVERSE_DIRECTION_PATTERNS = [re.compile(p) for p in (
    r"알려\s*주시겠",
    r"말씀해\s*주시",
    r"사례를\s*더",
    r"사례를\s*몇",
    r"설명해\s*주실\s*수\s*있나요",
    r"알려\s*주시면",
    r"공유해\s*주실",
    r"(?:관찰|언급|경험)하신",       # 읽는 사람(보호자)의 행동에 대한 존대 - 되묻는 방향
    r"있으신가요",
    r"보호자님",
    r"어머님|아버님",
)]


# ---------------------------------------------------------------- 텍스트 수집

def caregiver_texts(task: str, output: dict) -> list[tuple[str, str]]:
    """보호자 노출 생성 텍스트를 (블록, 텍스트)로 나열한다. 폴백·사전 요약 제외."""
    texts: list[tuple[str, str]] = []
    if task == "explain":
        for key in ("overview", "before_counseling"):
            if isinstance(output.get(key), str):
                texts.append((key, output[key]))
    else:
        for key, text_key in (("questions_for_counselor", "question"),
                              ("observation_points", "point")):
            for i, item in enumerate(output.get(key, [])):
                if isinstance(item, dict) and not item.get("_fallback") \
                        and isinstance(item.get(text_key), str):
                    texts.append((f"{key}[{i}]", item[text_key]))
    return texts


# ---------------------------------------------------------------- 지표 계산

def jargon_metrics(texts: list[tuple[str, str]]) -> dict:
    """(i) 전문 용어 잔존율."""
    term_hits = 0
    by_term: dict[str, int] = {}
    blocks_with_term = glossed = 0
    for _block, text in texts:
        hits_here = 0
        for name, pat in JARGON_TERMS.items():
            n = len(pat.findall(text))
            if n:
                by_term[name] = by_term.get(name, 0) + n
                hits_here += n
        if hits_here:
            blocks_with_term += 1
            term_hits += hits_here
            if any(p.search(text) for p in GLOSS_PATTERNS):
                glossed += 1
    return {
        "term_hits": term_hits,
        "by_term": dict(sorted(by_term.items(), key=lambda kv: -kv[1])),
        "blocks_total": len(texts),
        "blocks_with_term": blocks_with_term,
        "glossed_blocks": glossed,
        "residual_rate": (blocks_with_term / len(texts)) if texts else 0.0,
        "gloss_rate": (glossed / blocks_with_term) if blocks_with_term else None,
    }


def reflection_metrics(profile: CBCLProfile, prep_output: dict) -> dict:
    """(ii) 보호자 표현 반영률 (질문·관찰 항목 단위 + 토큰 단위)."""
    tokens = note_tokens(list(profile.caregiver_notes))
    items = caregiver_texts("prep", prep_output)
    reflected = 0
    hit_tokens: set[str] = set()
    per_item: list[tuple[str, list[str]]] = []
    for block, text in items:
        matched = [t for t in tokens if t in text]
        per_item.append((block, matched))
        if matched:
            reflected += 1
            hit_tokens |= set(matched)
    questions = [(b, m) for b, m in per_item if b.startswith("questions_for_counselor")]
    return {
        "tokens": tokens,
        "tokens_hit": sorted(hit_tokens, key=tokens.index),
        "token_rate": (len(hit_tokens) / len(tokens)) if tokens else None,
        "items_total": len(items),
        "items_reflected": reflected,
        "item_rate": (reflected / len(items)) if items else None,
        "questions_total": len(questions),
        "questions_reflected": sum(1 for _b, m in questions if m),
    }


def direction_warnings(prep_output: dict) -> list[dict]:
    """(iii) 상담사→보호자 방향으로 읽히는 질문 (WARN)."""
    warns = []
    for block, text in caregiver_texts("prep", prep_output):
        if not block.startswith("questions_for_counselor"):
            continue
        for pat in REVERSE_DIRECTION_PATTERNS:
            m = pat.search(text)
            if m:
                warns.append({"block": block, "matched": m.group(0), "question": text})
                break
    return warns


def quality_summary(profile: CBCLProfile, outputs: dict[str, dict]) -> dict:
    """프로파일 1건의 품질 지표 요약. outputs = {"explain": ..., "prep": ...}."""
    texts = caregiver_texts("explain", outputs.get("explain", {})) \
        + caregiver_texts("prep", outputs.get("prep", {}))
    prep = outputs.get("prep", {})
    return {
        "jargon": jargon_metrics(texts),
        "reflection": reflection_metrics(profile, prep),
        "direction_warnings": direction_warnings(prep),
    }


def fmt_rate(value) -> str:
    return "-" if value is None else f"{100.0 * value:.0f}%"
