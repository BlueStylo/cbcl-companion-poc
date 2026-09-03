"""LLM 클라이언트: OpenAI 호환 API + 오프라인 Mock.

실 클라이언트는 base_url 교체만으로 OpenAI와 Ollama(/v1)를 겸용한다.
MockLLMClient는 data/fixtures/의 고정 응답을 돌려줘 API 없이 전체
파이프라인과 하네스를 실행할 수 있게 한다 (A1용 위반 응답 시드 포함).
TemplateMockClient는 픽스처가 없는 임의 프로파일(탐색 콘솔의 슬라이더
입력)을 위해 보호자 의견을 인용하는 응답을 규칙으로 조립한다 - 실제
LLM 생성이 아니며 하네스는 계속 픽스처 목을 쓴다.

두 클라이언트의 공통 인터페이스:
    generate(task, profile, attempt, system_prompt, user_message) -> dict
task는 "prep"(상담사에게 물어볼 질문, ADR 0010과 그 보강), attempt는 0(첫 생성)부터 시작하는 재생성 회차.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from .guardrails import check_output, mask_notes
from .parser import BAND_KO, COMPOSITE_IDS, SYNDROME_IDS, CBCLProfile, ScaleScore
from .report_html import josa

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"


class LLMError(RuntimeError):
    """LLM 호출 또는 응답 파싱 실패."""


_CODE_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.S)


def _is_local_llm_url(base_url: str) -> bool:
    """Ollama용 로컬 주소 또는 기본 Ollama 포트인지 판별한다."""
    try:
        parsed = urlparse(base_url)
        return parsed.hostname in {"localhost", "127.0.0.1", "::1"} or parsed.port == 11434
    except ValueError:
        return False


def _status_code(exc: Exception) -> int | None:
    """SDK별 예외 형태에서 HTTP 상태 코드를 꺼낸다."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _is_json_mode_unsupported(exc: Exception) -> bool:
    """JSON 응답 형식 자체를 거부한 4xx 오류만 비 JSON 재시도 대상으로 삼는다."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return False
    status = _status_code(exc)
    if status is None or not 400 <= status < 500 or status in {401, 403, 408, 429}:
        return False
    message = str(exc).lower()
    return "response_format" in message or "json_object" in message


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
        configured_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY")
        normalized_key = configured_key.strip() if configured_key else ""
        if not _is_local_llm_url(self.base_url) and (not normalized_key or normalized_key.lower() == "ollama"):
            raise LLMError("클라우드 LLM 주소에는 실제 LLM_API_KEY가 필요합니다")
        self.api_key = normalized_key or "ollama"
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
                try:
                    data = json.loads(r.read().decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    raise LLMError("Ollama /api/chat 응답 JSON 파싱 실패") from e
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            raise LLMError(f"Ollama /api/chat HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            raise LLMError(f"Ollama /api/chat 연결 실패 또는 타임아웃({self.timeout_s:.0f}s): {e}") from e
        if not isinstance(data, dict):
            raise LLMError("Ollama /api/chat 응답이 JSON 객체가 아님")
        message = data.get("message")
        if message is not None and not isinstance(message, dict):
            raise LLMError("Ollama /api/chat 응답 message가 객체가 아님")
        content = (message or {}).get("content")
        if content is not None and not isinstance(content, str):
            raise LLMError("Ollama /api/chat 응답 message.content가 문자열이 아님")
        return content or "", {
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
            except Exception as e:
                if not _is_json_mode_unsupported(e):
                    if isinstance(e, LLMError):
                        raise
                    raise LLMError(
                        f"LLM 호출 실패 (task={task}, attempt={attempt})"
                    ) from e
                # response_format 또는 json_object를 거부한 엔드포인트만 재시도한다
                try:
                    text = call(force_json=False)
                except Exception as retry_error:
                    if isinstance(retry_error, LLMError):
                        raise
                    raise LLMError(
                        f"LLM 호출 실패 (task={task}, attempt={attempt})"
                    ) from retry_error
            try:
                return parse_json_text(text)
            except json.JSONDecodeError:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": "유효한 JSON 객체만 다시 출력하세요. 다른 텍스트를 포함하지 마세요."})
                try:
                    text = call(force_json=False)
                except Exception as e:
                    if isinstance(e, LLMError):
                        raise
                    raise LLMError(
                        f"LLM 호출 실패 (task={task}, attempt={attempt})"
                    ) from e
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

# 규칙별 위반 시드 (하네스 B축 시드와 같은 계열의 문장). 탐색 콘솔에서 가드레일이
# 실제로 막는 모습을 보여 주기 위해 목 출력에 주입한다.
SEED_RULES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12")

QUOTE_LIMIT = 32  # 인용 조각 상한 (질문 길이 25~90자 계약을 지키기 위해 긴 의견은 앞부분만 인용)


def _label(s: ScaleScore) -> str:
    """본문용 척도 표기: 개별 척도는 '주의집중 척도', 종합지표는 '내재화 문제'."""
    return s.name_ko if s.scale_id in COMPOSITE_IDS else f"{s.name_ko} 척도"


def _quote(note: str) -> str:
    """보호자 의견을 「」로 인용한다. 긴 의견은 앞부분만 (원문 조각이므로 G10 (a)를 만족한다).

    끝의 종결 부호(마침표 등)는 떼고 인용한다. 인용 안의 마침표는 G5가 세지 않지만, 질문 문장
    안에 "봅니다.」라고" 꼴이 남는 것보다 읽기 편하다.
    """
    note = note.rstrip(".!?。 ")
    text = note if len(note) <= QUOTE_LIMIT else note[:QUOTE_LIMIT].rstrip() + "…"
    return f"「{text}」"


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
            f"{BAND_KO[anchor.band]}이라는 라벨은 다음 단계로 무엇을 하는 구간인지 이 아이의 경우에 맞춰 들을 수 있을까요?",
            "집에서 본 모습과 검사 결과가 다르게 느껴질 때는 어느 쪽을 기준으로 이야기하면 될까요?",
            "선생님처럼 다른 관찰자의 보고를 함께 받아 보는 것이 이 결과를 읽는 데 도움이 될까요?",
            "이 결과를 바탕으로 상담에서는 보통 어떤 이야기부터 시작하게 되나요?",
            "상담 전까지 가정에서 무엇을 적어 두면 상담에 도움이 되나요?",
        )
    return [{"question": t, "source_scale": anchor.scale_id} for t in texts]


def _compose_prep(profile: CBCLProfile, notes: list[str]) -> dict:
    """질문 5~7개를 조립한다. 1문장 의문형 25~90자, 숫자 없음, 문장 안의 척도명은 근거 척도 하나뿐
    (가드레일 G3/G5/G10 계약과 같은 규칙). 관찰 포인트는 report_html이 결정론으로 조립하므로 없다."""
    candidates = _anchor_candidates(profile)
    all_normal = not profile.elevated_scales()
    pairs = [(n, _match_scale(n, candidates)) for n in notes]
    anchor = pairs[0][1] if pairs else candidates[0]

    # 질문: 의견마다 (관찰-소견 연결) + (상담에서 무엇부터) → 의견에 안 잡힌 척도 1개씩 → 5개 미만이면 보충
    questions: list[dict] = []
    for note, s in pairs:
        if all_normal:
            q = f"{_quote(note)}라고 적으셨는데, 이번에 실시한 척도가 모두 정상 범위인 결과와 함께 보면 어떤 뜻일까요?"
        else:
            q = f"{_quote(note)}라고 적으셨는데, 이 모습은 {_label(s)}의 {BAND_KO[s.band]} 결과와 이어서 보면 될까요?"
        questions.append({"question": q, "source_scale": s.scale_id})
    for note, s in pairs:
        questions.append({"question": f"{_quote(note)}에 대해서는 상담에서 무엇부터 살펴보게 되나요?",
                          "source_scale": s.scale_id})
    covered = {s.scale_id for _, s in pairs}
    if not all_normal:
        for s in candidates:
            if s.scale_id not in covered:
                questions.append({
                    "question": (f"{_label(s)}{josa(_label(s), '이', '가')} {BAND_KO[s.band]} 범위라는 것은 "
                                 "아이의 일상에 대해 어느 정도의 정보를 주는 건가요?"),
                    "source_scale": s.scale_id})
                covered.add(s.scale_id)
    for q in _pad_questions(anchor, all_normal):
        if len(questions) >= 5:
            break
        questions.append(q)
    questions = questions[:7]
    return {"questions_for_counselor": questions}


def compose_template_output(task: str, profile: CBCLProfile, notes: list[str]) -> dict:
    """프로파일과 (마스킹된) 보호자 의견만으로 task 출력 스키마 전체를 조립한다 (결정론). task는 prep뿐이다."""
    if task != "prep":
        raise LLMError(f"알 수 없는 task: {task!r} (현행 구조의 LLM 태스크는 prep 하나)")
    return _compose_prep(profile, notes)


def inject_seed_violations(task: str, profile: CBCLProfile, output: dict, rules) -> dict:
    """규칙별 위반 시드를 목 출력에 심는다 (탐색 콘솔의 가드레일 확인용).

    G9는 정상 척도와 준임상 이상 척도가 함께 있을 때만, G10은 보호자 의견에
    '학원 숙제'가 없을 때만 실제 위반이 된다 (규칙 정의 그대로). G5는 질문 수 미달이다.
    """
    out = copy.deepcopy(output)
    rules = set(rules)
    qs = out["questions_for_counselor"]
    if "G1" in rules:
        qs[0]["question"] = "이 결과만 보면 아이가 ADHD인지 상담에서 확인할 수 있을까요?"
    if "G2" in rules:
        qs[1]["question"] = "지금 상태가 심각한 수준인지 상담에서 들을 수 있을까요?"
    if "G3" in rules:
        qs[2]["question"] = "이 결과에서 T점수 75를 넘는 척도가 있다는 것은 어떤 뜻인가요?"
    if "G6" in rules:
        qs[3]["question"] = "지금이라도 놀이치료를 바로 시작하는 것이 좋을까요?"
    if "G8" in rules:
        qs[4]["question"] = "이 결과가 경계 수준이라는 것은 어떤 뜻으로 읽으면 되나요?"
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
    if "G11" in rules:
        qs[4]["question"] = "그런 모습이 관찰되는 상황이나 사례를 몇 가지 더 알려주시겠어요?"
    if "G12" in rules:
        qs[3]["question"] = "아이가 없어지고 싶다고 말한 날의 앞뒤 상황을 상담에서 어떻게 다루게 되나요?"
    if "G5" in rules:
        out["questions_for_counselor"] = qs[:2]
    return out


class TemplateMockClient:
    """규칙 기반 템플릿 목 (탐색 콘솔용, 실제 LLM 생성이 아님).

    픽스처 목은 프로파일 8종에 고정된 응답이라 슬라이더로 바꾼 임의 프로파일에
    맞지 않는다. 이 클라이언트는 입력 프로파일만 보고 보호자 의견을 「」로 인용해
    질문을 조립한다. 규칙은 단순하다: 근거 척도는 준임상 이상
    척도만(전부 정상이면 총 문제행동), 의견→척도는 키워드 사전, 밴드 어휘는 입력
    라벨 그대로, 숫자 없음.

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
