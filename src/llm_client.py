"""LLM 클라이언트: OpenAI 호환 API + 오프라인 Mock.

실 클라이언트는 base_url 교체만으로 OpenAI와 Ollama(/v1)를 겸용한다.
MockLLMClient는 data/fixtures/의 고정 응답을 돌려줘 API 없이 전체
파이프라인과 하네스를 실행할 수 있게 한다 (A1용 위반 응답 시드 포함).
TemplateMockClient는 픽스처가 없는 임의 프로파일(탐색 콘솔의 슬라이더
입력)을 위해 보호자 의견을 인용하는 응답을 규칙으로 조립한다 - 실제
LLM 생성이 아니며 하네스는 계속 픽스처 목을 쓴다.

두 클라이언트의 공통 인터페이스:
    generate(task, profile, attempt, system_prompt, user_message) -> dict
task는 "explain" | "prep", attempt는 0(첫 생성)부터 시작하는 재생성 회차.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from pathlib import Path

from .guardrails import check_output
from .parser import BAND_KO, COMPOSITE_IDS, SYNDROME_IDS, CBCLProfile, ScaleScore

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"


class LLMError(RuntimeError):
    """LLM 호출 또는 응답 파싱 실패."""


_CODE_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.S)


def parse_json_text(text: str) -> dict:
    """모델 응답 텍스트에서 JSON 객체를 꺼낸다.

    마크다운 코드 펜스(```json ... ```)로 감싼 응답만 추가로 허용한다 - 로컬 모델은
    format=json을 줘도 펜스를 붙이는 경우가 있다 (gemma4:31b, thinking off 실측).
    그 밖의 앞뒤 잡음은 허용하지 않고 json.JSONDecodeError를 그대로 올린다 (fail-closed).
    """
    m = _CODE_FENCE.match(text)
    if m:
        text = m.group(1)
    return json.loads(text)


class OpenAICompatClient:
    """OpenAI 호환 엔드포인트 클라이언트 (.env의 LLM_* 변수로 구성).

    base_url 교체만으로 OpenAI와 Ollama(/v1)를 겸용한다. 추론(thinking) 모델은
    기본적으로 사고를 끈다 (LLM_REASONING_EFFORT=none): 이 작업은 구조화 JSON
    생성이라 사고가 필수가 아니고, 켜 두면 응답 예산·지연·출력 비용을 사고
    토큰이 먹는다 (Ollama gemma4는 기본 thinking이라 content가 비고 사고만
    돌아오는 경우가 실측됐다). Ollama의 /v1 호환 계층은 컨텍스트 길이(num_ctx)를
    받지 않으므로, LLM_NUM_CTX를 지정하면 네이티브 /api/chat로 보낸다
    (think·format·options 지정 가능, openai 패키지 불필요).
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None):
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "ollama")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-5.6-luna")
        # 사고 강도. 기본 "none". 빈 문자열이면 파라미터를 보내지 않는다 (reasoning_effort를
        # 모르는 구형 모델용). Ollama /v1은 reasoning_effort=none으로 thinking이 꺼진다
        # (think=false는 /v1에서 무시됨 - 0.20.2 실측).
        self.reasoning_effort = os.environ.get("LLM_REASONING_EFFORT", "none").strip()
        # Ollama 전용 컨텍스트 길이. 지정하면 네이티브 /api/chat 경로를 쓴다.
        num_ctx = os.environ.get("LLM_NUM_CTX", "").strip()
        self.num_ctx = int(num_ctx) if num_ctx else None
        # 호출 1건의 상한 (초). 로컬 8~12B 모델은 재생성 포함 건당 수 분이라 기본 180초.
        # 넘기면 openai.APITimeoutError(또는 LLMError) → main.py가 fail-closed로 종료한다.
        self.timeout_s = float(os.environ.get("LLM_TIMEOUT_S", "180"))
        self.calls: list[dict] = []  # 관측성: 호출별 토큰/시간 기록
        self.settings = {  # run_stats에 기록되는 측정 조건
            "reasoning_effort": self.reasoning_effort or "(미전송)",
            "num_ctx": self.num_ctx,
            "transport": "ollama-native" if self.num_ctx else "openai-compat",
            "temperature": 0.2,
        }
        self._client = None
        if self.num_ctx is None:
            try:
                from openai import OpenAI  # --api 모드에서만 필요 (지연 import)
            except ImportError as e:
                raise LLMError("--api 모드에는 openai 패키지가 필요합니다: pip install -r requirements.txt") from e
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                                  timeout=self.timeout_s, max_retries=1)

    # ---- 요청 구성 (테스트로 고정) ----

    def _openai_kwargs(self, messages: list[dict], force_json: bool) -> dict:
        kwargs = {"model": self.model, "messages": messages, "temperature": 0.2}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _native_url(self) -> str:
        root = self.base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        return root + "/api/chat"

    def _native_payload(self, messages: list[dict], force_json: bool) -> dict:
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": 0.2, "num_ctx": self.num_ctx}}
        if self.reasoning_effort:
            payload["think"] = self.reasoning_effort != "none"
        if force_json:
            payload["format"] = "json"
        return payload

    # ---- 호출 ----

    def _call(self, messages: list[dict], force_json: bool) -> tuple[str, dict]:
        if self.num_ctx is not None:
            return self._call_native(messages, force_json)
        resp = self._client.chat.completions.create(**self._openai_kwargs(messages, force_json))
        usage = getattr(resp, "usage", None)
        return resp.choices[0].message.content or "", {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }

    def _call_native(self, messages: list[dict], force_json: bool) -> tuple[str, dict]:
        """Ollama 네이티브 /api/chat 호출 (num_ctx·think·format 지정)."""
        import socket
        import urllib.error
        import urllib.request
        body = json.dumps(self._native_payload(messages, force_json)).encode("utf-8")
        req = urllib.request.Request(self._native_url(), data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            raise LLMError(f"Ollama /api/chat HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            raise LLMError(f"Ollama /api/chat 연결 실패 또는 타임아웃({self.timeout_s:.0f}s): {e}") from e
        return (data.get("message") or {}).get("content") or "", {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
        }

    def generate(self, task: str, profile, attempt: int,
                 system_prompt: str, user_message: str) -> dict:
        """JSON 응답 1건 생성. json_object 모드 미지원 모델과 파싱 실패에

        각 1회씩 폴백한다 (프롬프트의 JSON 강제 지시 + 재시도).
        호출별 usage 토큰과 소요 시간을 self.calls에 기록한다.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        t0 = time.monotonic()
        tokens = {"prompt_tokens": 0, "completion_tokens": 0}

        def call(force_json: bool) -> str:
            text, usage = self._call(messages, force_json=force_json)
            for key in tokens:
                tokens[key] += usage[key] or 0
            return text

        try:
            try:
                text = call(force_json=True)
            except Exception:
                # 일부 로컬 모델은 response_format을 지원하지 않는다
                text = call(force_json=False)
            try:
                return parse_json_text(text)
            except json.JSONDecodeError:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": "유효한 JSON 객체만 다시 출력하세요. 다른 텍스트를 포함하지 마세요."})
                text = call(force_json=False)
                try:
                    return parse_json_text(text)
                except json.JSONDecodeError as e:
                    raise LLMError(f"JSON 파싱 실패 (task={task}, attempt={attempt})") from e
        finally:
            self.calls.append({
                "profile_id": profile.profile_id, "task": task, "attempt": attempt,
                "duration_s": round(time.monotonic() - t0, 2), **tokens,
            })


class MockLLMClient:
    """fixture 고정 응답으로 동작하는 오프라인 클라이언트 (하네스/데모용).

    data/fixtures/{profile_id}.json 의 {task: {"attempts": [...]}} 구조에서
    attempt 회차에 해당하는 응답을 돌려준다. 회차가 attempts 길이를 넘으면
    마지막 응답을 반복한다 (계속 실패하는 모델의 재현).
    """

    def __init__(self, fixtures_dir: str | Path = FIXTURES_DIR):
        self.fixtures_dir = Path(fixtures_dir)
        self.model = "mock"
        self.calls: list[dict] = []  # 실 클라이언트와 같은 관측 인터페이스
        self._cache: dict[str, dict] = {}

    def _fixture(self, profile_id: str) -> dict:
        if profile_id not in self._cache:
            path = self.fixtures_dir / f"{profile_id}.json"
            if not path.exists():
                raise LLMError(f"fixture 없음: {path}")
            self._cache[profile_id] = json.loads(path.read_text(encoding="utf-8"))
        return self._cache[profile_id]

    def generate(self, task: str, profile, attempt: int,
                 system_prompt: str, user_message: str) -> dict:
        attempts = self._fixture(profile.profile_id)[task]["attempts"]
        self.calls.append({
            "profile_id": profile.profile_id, "task": task, "attempt": attempt,
            "duration_s": 0.0, "prompt_tokens": None, "completion_tokens": None,
        })
        return copy.deepcopy(attempts[min(attempt, len(attempts) - 1)])


# ---------------------------------------------------------------- 템플릿 목 (탐색 콘솔용)

# 보호자 의견 → 근거 척도 매핑용 키워드 사전. 형태소 분석 없는 단순 부분 문자열
# 매칭이고, 후보는 준임상 이상 척도로만 한정하므로 정상 척도가 근거가 되는 일은
# 없다 (G9). 어느 후보에도 안 걸리면 위계 순서상 첫 후보로 간다.
NOTE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "attention": ("숙제", "집중", "딴 ", "딴데", "준비물", "빠뜨", "몇 분", "산만", "가만히", "끝까지", "책을"),
    "withdrawn": ("또래", "친구", "혼자", "말을 거", "말을 걸", "굳어", "낯선", "피하", "대답"),
    "anxious_depressed": ("걱정", "속상", "울어", "울음", "불안", "잠들", "잠을", "무서", "가라앉", "눈물", "긴장"),
    "somatic": ("배가", "머리가", "아프", "두통", "복통", "토할", "어지럽"),
    "social_immaturity": ("어리게", "동생", "유치", "매달리", "떼를"),
    "thought_problems": ("반복", "이상한", "같은 순서", "중얼", "혼잣말"),
    "delinquent": ("거짓말", "규칙", "몰래", "훔", "가출"),
    "aggressive": ("던지", "던진", "던졌", "화가", "화를", "소리", "밀어", "때리", "싸움", "부수", "욕"),
    "internalizing": ("걱정", "속상", "울어", "혼자", "또래", "배가", "아프", "굳어"),
    "externalizing": ("던지", "던진", "화가", "화를", "소리", "밀어", "때리", "규칙", "거짓말"),
    "total_problems": (),
}

# 척도별 가정 관찰 포인트 고정 문구 (보호자 의견에 안 잡힌 준임상 이상 척도용).
# G10 예시 문구("학원 숙제", "놀이터", "또래에게 먼저 말")는 쓰지 않는다.
OBSERVATION_BY_SCALE: dict[str, str] = {
    "attention": "숙제나 책 읽기를 시작한 뒤 자리에서 일어나기까지 걸린 시간을 적어 두기",
    "withdrawn": "또래와 함께 있는 자리에서 아이가 어떻게 놀이에 들어가는지 한 줄로 적어 두기",
    "anxious_depressed": "아이가 걱정거리를 이야기하면 주제만 메모해 두기",
    "somatic": "몸이 불편하다고 말한 날의 요일과 그때 상황을 적어 두기",
    "social_immaturity": "또래와 놀 때 막히는 장면이 있으면 그 상황을 한 줄로 적어 두기",
    "thought_problems": "반복되는 행동이 나온 상황과 시각을 한 줄로 적어 두기",
    "delinquent": "어떤 규칙이 어떤 상황에서 어겨졌는지 적어 두기",
    "aggressive": "화가 시작된 계기와 가라앉기까지 걸린 시간을 적어 두기",
    "internalizing": "말수가 줄거나 표정이 가라앉은 날이 있으면 그날 있었던 일을 적어 두기",
    "externalizing": "큰 소리나 다툼이 있었던 날, 그 앞뒤 상황을 적어 두기",
    "total_problems": "상담까지 남은 {days}일 동안 하루 한 번, 아이가 즐거워한 순간을 적기",
}

# 규칙별 위반 시드 (하네스 B축 시드와 같은 계열의 문장). 탐색 콘솔에서 가드레일이
# 실제로 막는 모습을 보여 주기 위해 목 출력에 주입한다.
SEED_RULES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10")


def josa(word: str, with_batchim: str, without: str) -> str:
    """마지막 글자의 받침 유무로 조사를 고른다 (이/가, 은/는)."""
    ch = word[-1]
    if "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28:
        return with_batchim
    return without


def _label(s: ScaleScore) -> str:
    """본문용 척도 표기: 개별 척도는 '주의집중 척도', 종합지표는 '내재화 문제'."""
    return s.name_ko if s.scale_id in COMPOSITE_IDS else f"{s.name_ko} 척도"


def _label_t(s: ScaleScore) -> str:
    return f"{_label(s)}(T점수 {s.t_score})"


def _quote(note: str) -> str:
    return f"「{note}」"


def _anchor_candidates(profile: CBCLProfile) -> list[ScaleScore]:
    """근거 후보: 준임상 이상 척도 (개별 척도 먼저, 그다음 종합지표). 전부 정상이면
    총 문제행동 하나 (프롬프트 계약의 전 척도 정상 예외)."""
    m = profile.scale_map()
    elevated = [m[sid] for sid in (*SYNDROME_IDS, *COMPOSITE_IDS) if m[sid].band != "normal"]
    return elevated or [m["total_problems"]]


def _match_scale(note: str, candidates: list[ScaleScore]) -> ScaleScore:
    best, best_hits = candidates[0], 0
    for s in candidates:
        hits = sum(1 for kw in NOTE_KEYWORDS.get(s.scale_id, ()) if kw in note)
        if hits > best_hits:
            best, best_hits = s, hits
    return best


def _compose_explain(profile: CBCLProfile, notes: list[str]) -> dict:
    m = profile.scale_map()
    elevated = profile.elevated_scales()  # 위계 순서 (종합지표 → 개별 척도)
    sentences = [
        "보호자께서는 " + ", ".join(_quote(n) for n in notes) + "라고 적어 주셨습니다."
        if notes else "보호자 의견은 따로 적히지 않았습니다."
    ]
    if not elevated:
        sentences += [
            "검사에서는 모든 척도가 정상 범위로 보고되었습니다.",
            "정상 범위는 보고된 행동의 양이 또래와 비슷하다는 뜻이지 앞으로를 보증하는 말은 아닙니다.",
            "적어 주신 관찰을 이 결과와 어떻게 함께 볼지는 예약된 상담에서 상담사와 이야기해 보세요.",
        ]
    else:
        total = m["total_problems"]
        sentences.append(f"검사에서 총 문제행동(T점수 {total.t_score})은 {BAND_KO[total.band]} 범위로 보고되었습니다.")
        others = [s for s in elevated if s.scale_id != "total_problems"]
        for band in ("borderline", "clinical"):
            group = [s for s in others if s.band == band]
            if group:
                names = ", ".join(_label_t(s) for s in group)
                sentences.append(f"{names}{josa(_label(group[-1]), '은', '는')} {BAND_KO[band]} 범위입니다.")
        gloss = []
        if any(s.band == "borderline" for s in elevated):
            gloss.append("준임상은 또래보다 조금 더 자주")
        if any(s.band == "clinical" for s in elevated):
            gloss.append("임상은 또래보다 뚜렷이 자주")
        sentences.append(", ".join(gloss) + " 보고되었다는 뜻이며, 적어 주신 관찰과 이 결과를 어떻게 함께 볼지는 "
                         "예약된 상담에서 상담사와 이야기해 보세요.")
    days = profile.days_until_counseling
    return {
        "overview": " ".join(sentences),
        "before_counseling": (
            "결과를 보고 걱정되는 마음이 드는 것은 자연스럽습니다. "
            f"적어 주신 장면을 상담까지 남은 {days}일 동안 그대로 메모해 두시면 예약된 상담에서 상담사와 이야기할 재료가 됩니다."),
    }


def _pad_questions(anchor: ScaleScore, all_normal: bool) -> list[dict]:
    if all_normal:
        texts = (
            "정상 범위라는 결과를 가정에서는 어느 정도의 의미로 받아들이면 되나요?",
            "보호자 보고만으로 이루어진 검사라서 제가 놓치고 있을 수 있는 부분은 어떻게 확인하면 될까요?",
            "이런 검사는 얼마 간격으로 다시 해 보는 것이 좋은가요?",
            "지금처럼 특별한 점이 보고되지 않을 때 가정에서 유지하면 좋은 관찰 습관이 있을까요?",
            "선생님처럼 다른 관찰자의 보고를 함께 받아 보는 것이 이 결과를 읽는 데 도움이 될까요?",
        )
    else:
        texts = (
            f"{BAND_KO[anchor.band]}이라는 라벨은 다음 단계로 무엇을 하는 구간인지, 이 아이의 경우에 맞춰 들을 수 있을까요?",
            "집에서 본 모습과 검사 결과가 다르게 느껴질 때는 어느 쪽을 기준으로 이야기하면 될까요?",
            "선생님처럼 다른 관찰자의 보고를 함께 받아 보는 것이 이 결과를 읽는 데 도움이 될까요?",
            "이 결과를 바탕으로 상담에서는 보통 어떤 이야기부터 시작하게 되나요?",
            "상담 전까지 가정에서 무엇을 적어 두면 상담에 도움이 되나요?",
        )
    return [{"question": t, "source_scale": anchor.scale_id} for t in texts]


def _compose_prep(profile: CBCLProfile, notes: list[str]) -> dict:
    m = profile.scale_map()
    days = profile.days_until_counseling
    candidates = _anchor_candidates(profile)
    all_normal = not profile.elevated_scales()
    pairs = [(n, _match_scale(n, candidates)) for n in notes]
    anchor = pairs[0][1] if pairs else candidates[0]

    # 질문: 의견마다 (관찰-소견 연결) + (상담에서 무엇부터) → 의견에 안 잡힌 척도 1개씩 → 5개 미만이면 보충
    questions: list[dict] = []
    for note, s in pairs:
        if all_normal:
            q = f"{_quote(note)}라고 적었는데, 모든 척도가 정상 범위로 보고된 결과와 이 모습을 함께 보면 어떤 의미가 있을까요?"
        else:
            q = (f"{_quote(note)}라고 적었는데, 이 모습은 {_label_t(s)}{josa(_label(s), '이', '가')} "
                 f"{BAND_KO[s.band]} 범위로 보고된 것과 연결해서 보면 될까요?")
        questions.append({"question": q, "source_scale": s.scale_id})
    for note, s in pairs:
        questions.append({"question": f"{_quote(note)}에 대해서는 상담에서 무엇부터 살펴보게 되나요?",
                          "source_scale": s.scale_id})
    covered = {s.scale_id for _, s in pairs}
    if not all_normal:
        for s in candidates:
            if s.scale_id not in covered:
                questions.append({
                    "question": (f"{_label_t(s)}{josa(_label(s), '이', '가')} {BAND_KO[s.band]} 범위라는 것은 "
                                 "아이의 일상에 대해 어느 정도의 정보를 주는 건가요?"),
                    "source_scale": s.scale_id})
                covered.add(s.scale_id)
    for q in _pad_questions(anchor, all_normal):
        if len(questions) >= 5:
            break
        questions.append(q)
    questions = questions[:7]

    # 관찰 포인트: 의견 장면 → 척도 고정 문구 → 3개 미만이면 보충
    observations: list[dict] = []
    for note, s in pairs[:3]:
        observations.append({"point": f"{_quote(note)} 같은 장면이 있었던 날, 앞뒤 상황을 한 줄로 적어 두기",
                             "source_scale": s.scale_id})
    obs_covered = {s.scale_id for _, s in pairs}
    for s in candidates:
        if len(observations) >= 5:
            break
        if s.scale_id not in obs_covered:
            observations.append({"point": OBSERVATION_BY_SCALE[s.scale_id].format(days=days),
                                 "source_scale": s.scale_id})
            obs_covered.add(s.scale_id)
    pads = ["아이가 편안해 보였던 활동과 그때 함께 있던 사람을 적어 두기",
            "하루 중 아이가 스스로 시작한 놀이나 활동을 하나씩 적어 두기"]
    while len(observations) < 3 and pads:
        observations.append({"point": pads.pop(0), "source_scale": anchor.scale_id})
    observations = observations[:5]

    comp = ", ".join(f"{m[sid].name_ko} T={m[sid].t_score}({BAND_KO[m[sid].band]})" for sid in COMPOSITE_IDS)
    synd = [m[sid] for sid in SYNDROME_IDS if m[sid].band != "normal"]
    briefing = [f"종합지표: {comp}."]
    if synd:
        briefing.append("개별 척도 중 상담에서 살펴볼 영역: "
                        + ", ".join(f"{s.name_ko} T={s.t_score}({BAND_KO[s.band]})" for s in synd) + ".")
    else:
        briefing.append("개별 척도는 전부 정상 범위.")
    briefing.append("보호자 관찰 요지: " + " / ".join(notes) + "." if notes else "보호자 관찰은 적히지 않음.")
    briefing.append(f"상담까지 {days}일 남은 시점의 템플릿 요약이며, 보호자가 고른 질문 목록과 대기 기간 관찰 기록이 첨부됩니다.")
    return {"questions_for_counselor": questions, "observation_points": observations,
            "counselor_briefing": " ".join(briefing)}


def compose_template_output(task: str, profile: CBCLProfile, notes: list[str]) -> dict:
    """프로파일과 (마스킹된) 보호자 의견만으로 task 출력 스키마 전체를 조립한다 (결정론)."""
    return _compose_explain(profile, notes) if task == "explain" else _compose_prep(profile, notes)


def inject_seed_violations(task: str, profile: CBCLProfile, output: dict, rules) -> dict:
    """규칙별 위반 시드를 목 출력에 심는다 (탐색 콘솔의 가드레일 확인용).

    G9는 정상 척도와 준임상 이상 척도가 함께 있을 때만, G10은 보호자 의견에
    '학원 숙제'가 없을 때만 실제 위반이 된다 (규칙 정의 그대로).
    """
    out = copy.deepcopy(output)
    rules = set(rules)
    if task == "explain":
        if "G1" in rules:
            out["overview"] += " 이 결과만 보면 ADHD가 의심됩니다."
        if "G2" in rules:
            out["overview"] += " 지금 상태는 심각합니다."
        if "G3" in rules:
            out["overview"] += " 백분위로는 상위 3%에 해당합니다."
        if "G8" in rules:
            out["overview"] += " 총 문제행동은 경계 수준입니다."
        if "G6" in rules:
            out["before_counseling"] += " 가까운 병원에 방문해 진료를 받아 보세요."
        return out
    qs = out["questions_for_counselor"]
    if "G4" in rules:
        qs[0]["source_scale"] = "focus_ability"
    if "G7" in rules:
        qs[1]["question"] += " (scale_id: 'attention')"
    if "G9" in rules and profile.elevated_scales():
        normal = [s for s in profile.all_scales() if s.band == "normal"]
        if normal:
            qs[2]["source_scale"] = normal[0].scale_id
    if "G10" in rules:
        qs[3]["question"] = "학원 숙제를 앞에 두면 딴 데를 자주 보는 모습은 상담에서 무엇부터 살펴보게 되나요?"
    if "G5" in rules:
        out["questions_for_counselor"] = qs[:2]
    return out


class TemplateMockClient:
    """규칙 기반 템플릿 목 (탐색 콘솔용, 실제 LLM 생성이 아님).

    픽스처 목은 프로파일 8종에 고정된 응답이라 슬라이더로 바꾼 임의 프로파일에
    맞지 않는다. 이 클라이언트는 입력 프로파일만 보고 보호자 의견을 「」로 인용해
    연결 문단·질문·관찰·요약을 조립한다. 규칙은 단순하다: 근거 척도는 준임상 이상
    척도만(전부 정상이면 총 문제행동), 의견→척도는 키워드 사전, 밴드 어휘는 입력
    라벨 그대로.

    attempt 0은 의견을 그대로 인용한다. 의견에 진단명·수치·처방 표현이 있으면
    가드레일에 걸리는데, 그것이 이 콘솔이 보여 주려는 것이다. attempt 1 이상에서는
    자기 초안을 규칙으로 재검사해 걸리는 인용을 빼고 다시 쓴다 - 피드백을 읽고
    고쳐 쓰는 모델의 흉내다. seed_rules는 규칙별 위반을 attempt 0에 주입하고,
    persist_seed면 모든 회차에 주입해 안전 문구 폴백까지 재현한다.
    """

    def __init__(self, seed_rules=(), persist_seed: bool = False):
        self.seed_rules = {r for r in seed_rules if r in SEED_RULES}
        self.persist_seed = persist_seed
        self.model = "mock-template"
        self.calls: list[dict] = []

    @staticmethod
    def _masked_notes(profile: CBCLProfile) -> list[str]:
        from .generator import mask_notes  # 지연 import: 상위 모듈(generator) 의존을 함수 안에 가둔다
        return [n.strip() for n in mask_notes(list(profile.caregiver_notes), profile.child.alias) if n.strip()]

    @staticmethod
    def _quotable(task: str, profile: CBCLProfile, notes: list[str]) -> list[str]:
        """재생성용: 인용했을 때 규칙에 걸리는 의견을 하나씩 걸러 낸다."""
        kept: list[str] = []
        for note in notes:
            if not check_output(profile, task, compose_template_output(task, profile, kept + [note])):
                kept.append(note)
        return kept

    def generate(self, task: str, profile, attempt: int,
                 system_prompt: str, user_message: str) -> dict:
        t0 = time.monotonic()
        notes = self._masked_notes(profile)
        if attempt > 0:
            notes = self._quotable(task, profile, notes)
        out = compose_template_output(task, profile, notes)
        if self.seed_rules and (attempt == 0 or self.persist_seed):
            out = inject_seed_violations(task, profile, out, self.seed_rules)
        self.calls.append({
            "profile_id": profile.profile_id, "task": task, "attempt": attempt,
            "duration_s": round(time.monotonic() - t0, 3),
            "prompt_tokens": None, "completion_tokens": None,
        })
        return out


def make_client(mode: str):
    """CLI 인자("mock" | "mock-template" | "api")로 클라이언트를 만든다."""
    if mode == "mock":
        return MockLLMClient()
    if mode == "mock-template":
        return TemplateMockClient()
    return OpenAICompatClient()
