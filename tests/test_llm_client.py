"""OpenAICompatClient 요청 구성 테스트 (네트워크 없음).

사고(thinking) 기본 off와 Ollama 네이티브 경로의 페이로드를 고정한다.
"""

import json
from types import SimpleNamespace

import pytest

from src.llm_client import LLMError, OpenAICompatClient, parse_json_text

ENV_KEYS = ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "LLM_REASONING_EFFORT",
            "LLM_NUM_CTX", "LLM_TIMEOUT_S")
MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def make(monkeypatch, with_default_key=True, **env):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    if with_default_key and "LLM_API_KEY" not in env:
        env["LLM_API_KEY"] = "test-key"
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


def _generate_with_fake_call(client, fake_call):
    client._call = fake_call
    return client.generate(
        task="prep",
        profile=SimpleNamespace(profile_id="test-profile"),
        attempt=0,
        system_prompt="system",
        user_message="user",
    )


def test_generate_returns_first_json_mode_success(monkeypatch):
    pytest.importorskip("openai")
    client = make(monkeypatch)
    calls = []

    def fake_call(_messages, force_json):
        calls.append(force_json)
        return '{"ok": true}', {"prompt_tokens": 1, "completion_tokens": 2}

    assert _generate_with_fake_call(client, fake_call) == {"ok": True}
    assert calls == [True]


def test_generate_retries_without_json_mode_only_for_format_error(monkeypatch):
    pytest.importorskip("openai")
    client = make(monkeypatch)
    calls = []

    class FormatError(RuntimeError):
        status_code = 400

    def fake_call(_messages, force_json):
        calls.append(force_json)
        if len(calls) == 1:
            raise FormatError("response_format json_object is not supported")
        return '{"ok": true}', {"prompt_tokens": 1, "completion_tokens": 2}

    assert _generate_with_fake_call(client, fake_call) == {"ok": True}
    assert calls == [True, False]


def test_generate_timeout_fails_after_one_low_level_call(monkeypatch):
    pytest.importorskip("openai")
    client = make(monkeypatch)
    calls = []

    def fake_call(_messages, force_json):
        calls.append(force_json)
        raise TimeoutError("timed out")

    with pytest.raises(LLMError, match="LLM 호출 실패"):
        _generate_with_fake_call(client, fake_call)
    assert calls == [True]


def test_generate_final_json_parse_failure_is_llm_error(monkeypatch):
    pytest.importorskip("openai")
    client = make(monkeypatch)
    calls = []

    def fake_call(_messages, force_json):
        calls.append(force_json)
        return "not json", {"prompt_tokens": 1, "completion_tokens": 2}

    with pytest.raises(LLMError, match="JSON 파싱 실패"):
        _generate_with_fake_call(client, fake_call)
    assert calls == [True, False]


def test_generate_format_fallback_final_call_failure_is_llm_error(monkeypatch):
    pytest.importorskip("openai")
    client = make(monkeypatch)
    calls = []

    class FormatError(RuntimeError):
        status_code = 400

    def fake_call(_messages, force_json):
        calls.append(force_json)
        if len(calls) == 1:
            raise FormatError("json_object response_format is not supported")
        raise TimeoutError("retry timed out")

    with pytest.raises(LLMError, match="LLM 호출 실패"):
        _generate_with_fake_call(client, fake_call)
    assert calls == [True, False]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ConnectionError("network down"), "LLM 호출 실패"),
        (type("AuthError", (RuntimeError,), {"status_code": 401})("authentication failed"), "LLM 호출 실패"),
        (type("BadRequest", (RuntimeError,), {"status_code": 400})("model is unavailable"), "LLM 호출 실패"),
    ],
)
def test_generate_non_format_failures_do_not_retry(monkeypatch, error, message):
    pytest.importorskip("openai")
    client = make(monkeypatch)
    calls = []

    def fake_call(_messages, force_json):
        calls.append(force_json)
        raise error

    with pytest.raises(LLMError, match=message):
        _generate_with_fake_call(client, fake_call)
    assert calls == [True]


@pytest.mark.parametrize("api_key", [None, "", "ollama", " OLLAMA "])
def test_remote_endpoint_requires_real_api_key(monkeypatch, api_key):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    if api_key is not None:
        monkeypatch.setenv("LLM_API_KEY", api_key)
    with pytest.raises(LLMError, match="실제 LLM_API_KEY"):
        OpenAICompatClient(base_url="https://api.openai.com/v1")


def test_local_ollama_endpoint_allows_dummy_key(monkeypatch):
    client = make(
        monkeypatch,
        with_default_key=False,
        LLM_BASE_URL="http://localhost:11434/v1",
        LLM_NUM_CTX="8192",
    )
    assert client.api_key == "ollama"


# ---------------------------------------------------------------- 템플릿 목의 인용

def test_template_mock_quotes_notes_ending_with_period_without_tripping_g5():
    """마침표로 끝나는 보호자 의견(흔한 입력)도 attempt 0에서 「」 인용이 살아남아야 한다.

    이전에는 _quote가 원문을 그대로 넣어 「...봅니다.」가 G5 1문장 검사에 걸리고, attempt 1이 인용을
    전부 버려 탐색 콘솔의 핵심 데모(보호자 표현 인용)가 조용히 사라졌다.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.generator import generate_all
    from src.guardrails import check_output
    from src.llm_client import TemplateMockClient
    from src.parser import load_profile

    base = load_profile(Path(__file__).resolve().parents[1] / "data/profiles/p2_partial_borderline.json")
    profile = base.model_copy(update={"caregiver_notes": ["학원 숙제를 앞에 두면 딴 데를 자주 봅니다.",
                                                          "놀이터에서 또래에게 먼저 말을 거는 일이 줄었습니다!"]})
    out = TemplateMockClient().generate("prep", profile, 0, "", "")
    assert check_output(profile, "prep", out) == []
    texts = [q["question"] for q in out["questions_for_counselor"]] + [o["point"] for o in out["observation_points"]]
    quoted = [t for t in texts if "「" in t]
    assert len(quoted) >= 4 and all("」" in t for t in quoted)
    assert not any(".」" in t or "!」" in t for t in texts)          # 종결 부호는 떼고 인용한다
    results = generate_all(profile, TemplateMockClient())
    assert results["prep"].regen_count == 0 and results["prep"].fallback_blocks == []
    assert sum("「" in q["question"] for q in results["prep"].output["questions_for_counselor"]) >= 2
