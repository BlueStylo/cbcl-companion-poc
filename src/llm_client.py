"""LLM 클라이언트: OpenAI 호환 API + 오프라인 Mock.

실 클라이언트는 base_url 교체만으로 OpenAI와 Ollama(/v1)를 겸용한다.
MockLLMClient는 data/fixtures/의 고정 응답을 돌려줘 API 없이 전체
파이프라인과 하네스를 실행할 수 있게 한다 (A1용 위반 응답 시드 포함).

두 클라이언트의 공통 인터페이스:
    generate(task, profile, attempt, system_prompt, user_message) -> dict
task는 "explain" | "prep", attempt는 0(첫 생성)부터 시작하는 재생성 회차.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"


class LLMError(RuntimeError):
    """LLM 호출 또는 응답 파싱 실패."""


class OpenAICompatClient:
    """OpenAI 호환 엔드포인트 클라이언트 (.env의 LLM_* 3개 변수로 구성)."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None):
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "ollama")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        try:
            from openai import OpenAI  # --api 모드에서만 필요 (지연 import)
        except ImportError as e:
            raise LLMError("--api 모드에는 openai 패키지가 필요합니다: pip install -r requirements.txt") from e
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _call(self, messages: list[dict], force_json: bool) -> str:
        kwargs = {"model": self.model, "messages": messages, "temperature": 0.2}
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def generate(self, task: str, profile, attempt: int,
                 system_prompt: str, user_message: str) -> dict:
        """JSON 응답 1건 생성. json_object 모드 미지원 모델과 파싱 실패에

        각 1회씩 폴백한다 (프롬프트의 JSON 강제 지시 + 재시도).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            text = self._call(messages, force_json=True)
        except Exception:
            # 일부 로컬 모델은 response_format을 지원하지 않는다
            text = self._call(messages, force_json=False)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "유효한 JSON 객체만 다시 출력하세요. 다른 텍스트를 포함하지 마세요."})
            text = self._call(messages, force_json=False)
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise LLMError(f"JSON 파싱 실패 (task={task}, attempt={attempt})") from e


class MockLLMClient:
    """fixture 고정 응답으로 동작하는 오프라인 클라이언트 (하네스/데모용).

    data/fixtures/{profile_id}.json 의 {task: {"attempts": [...]}} 구조에서
    attempt 회차에 해당하는 응답을 돌려준다. 회차가 attempts 길이를 넘으면
    마지막 응답을 반복한다 (계속 실패하는 모델의 재현).
    """

    def __init__(self, fixtures_dir: str | Path = FIXTURES_DIR):
        self.fixtures_dir = Path(fixtures_dir)
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
        return copy.deepcopy(attempts[min(attempt, len(attempts) - 1)])


def make_client(mode: str):
    """CLI 인자("mock" | "api")로 클라이언트를 만든다."""
    return MockLLMClient() if mode == "mock" else OpenAICompatClient()
