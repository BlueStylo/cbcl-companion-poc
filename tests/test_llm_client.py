"""OpenAICompatClient 요청 구성 테스트 (네트워크 없음).

사고(thinking) 기본 off와 Ollama 네이티브 경로의 페이로드를 고정한다.
"""

import pytest

import json

from src.llm_client import OpenAICompatClient, parse_json_text

ENV_KEYS = ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "LLM_REASONING_EFFORT",
            "LLM_NUM_CTX", "LLM_TIMEOUT_S")
MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def make(monkeypatch, **env):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return OpenAICompatClient()


def test_default_is_reasoning_off_with_json_mode(monkeypatch):
    pytest.importorskip("openai")
    c = make(monkeypatch)
    kwargs = c._openai_kwargs(MSGS, force_json=True)
    assert c.model == "gpt-5.6-luna"
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] == 0.2
    assert c.settings["transport"] == "openai-compat"


def test_empty_reasoning_effort_omits_param(monkeypatch):
    pytest.importorskip("openai")
    c = make(monkeypatch, LLM_REASONING_EFFORT="")
    assert "reasoning_effort" not in c._openai_kwargs(MSGS, force_json=False)
    assert c.settings["reasoning_effort"] == "(미전송)"


def test_num_ctx_routes_to_ollama_native_with_think_off(monkeypatch):
    c = make(monkeypatch, LLM_BASE_URL="http://localhost:11434/v1",
             LLM_MODEL="gemma4:12b", LLM_NUM_CTX="8192")
    assert c._native_url() == "http://localhost:11434/api/chat"
    payload = c._native_payload(MSGS, force_json=True)
    assert payload["think"] is False
    assert payload["format"] == "json"
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0.2, "num_ctx": 8192}
    assert c.settings["transport"] == "ollama-native"
    # force_json=False면 format을 보내지 않는다 (JSON 재요청 경로)
    assert "format" not in c._native_payload(MSGS, force_json=False)


def test_parse_json_text_accepts_markdown_fence_only():
    assert parse_json_text('{"a": 1}') == {"a": 1}
    assert parse_json_text('```json\n{\n  "a": 1\n}\n```') == {"a": 1}
    assert parse_json_text('  ```\n{"a": [1, 2]}\n```  ') == {"a": [1, 2]}
    with pytest.raises(json.JSONDecodeError):
        parse_json_text('여기 결과입니다: {"a": 1}')
    with pytest.raises(json.JSONDecodeError):
        parse_json_text('```json\n{"a": 1}\n``` 끝')
