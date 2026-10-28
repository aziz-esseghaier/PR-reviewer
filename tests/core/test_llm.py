"""Tests for the LLM client, run against a fake provider so they use no API quota."""

import json
from collections.abc import Callable
from typing import Any

import httpx2
import openai
import pytest

from pr_reviewer.core.config import ConfigError, ModelProfile, ModelsConfig, ProviderConfig
from pr_reviewer.core.llm import LLMClient, explain_api_error, list_provider_models

Handler = Callable[[httpx2.Request], httpx2.Response]

CONFIG = ModelsConfig(
    providers={
        "fake": ProviderConfig(base_url="https://llm.example.test/v1", api_key_env="FAKE_LLM_KEY")
    },
    models={
        "fake-small": ModelProfile(
            provider="fake", model="fake-model-v1", temperature=0.0, max_tokens=64
        )
    },
)


def completion_body(content: str | None, finish_reason: str = "stop") -> dict[str, Any]:
    """Build a chat completion response the way OpenAI-compatible providers send it.

    Args:
        content (str | None): The reply text. None mimics a model that wrote nothing.
        finish_reason (str): Why the model stopped.

    Returns:
        dict[str, Any]: The JSON body.
    """
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "fake-model-v1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }


def fake_client(handler: Handler) -> LLMClient:
    """Create a client for the fake profile whose requests go to a handler, not the network.

    Args:
        handler (Handler): Function that receives each request and returns the response.

    Returns:
        LLMClient: The client.
    """
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return LLMClient("fake-small", CONFIG, api_key="test-key", http_client=http)


def test_complete_sends_profile_settings_and_reads_reply() -> None:
    """Check the request carries the profile's model and settings, and the reply is parsed."""
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=completion_body("Hi there."))

    response = fake_client(handler).complete(
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}]
    )

    request = seen[0]
    assert str(request.url) == "https://llm.example.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == "fake-model-v1"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 64
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
    ]
    assert response.text == "Hi there."
    assert response.profile == "fake-small"
    assert response.provider == "fake"
    assert response.finish_reason == "stop"
    assert (response.input_tokens, response.output_tokens) == (12, 3)
    assert response.latency_s >= 0


def test_empty_reply_becomes_empty_text() -> None:
    """Check that a reply cut off before any text gives "" and finish reason "length"."""
    client = fake_client(lambda request: httpx2.Response(200, json=completion_body(None, "length")))
    response = client.complete([{"role": "user", "content": "Hello"}])
    assert response.text == ""
    assert response.finish_reason == "length"


def test_rate_limit_is_raised_without_hidden_retries() -> None:
    """Check that a 429 reaches our code after exactly one request, with no SDK retries."""
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(429, json={"error": {"message": "Rate limit reached"}})

    with pytest.raises(openai.RateLimitError) as exc_info:
        fake_client(handler).complete([{"role": "user", "content": "Hello"}])
    assert calls == 1
    assert "Wait a bit" in explain_api_error(exc_info.value)


def test_missing_api_key_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that creating a client without its key fails early and says which variable is empty."""
    monkeypatch.delenv("FAKE_LLM_KEY", raising=False)
    with pytest.raises(ConfigError, match="FAKE_LLM_KEY"):
        LLMClient("fake-small", CONFIG)


def test_unknown_profile_is_rejected() -> None:
    """Check that asking for a profile that isn't configured fails before any request."""
    with pytest.raises(ConfigError, match="No model profile"):
        LLMClient("missing", CONFIG, api_key="test-key")


def test_list_provider_models_returns_sorted_ids() -> None:
    """Check that the provider's model list comes back as sorted IDs."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/models"
        models = [{"id": name, "object": "model", "created": 0, "owned_by": "x"} for name in "ba"]
        return httpx2.Response(200, json={"object": "list", "data": models})

    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    provider = CONFIG.provider("fake")
    assert list_provider_models(provider, api_key="test-key", http_client=http) == ["a", "b"]


@pytest.mark.parametrize(
    ("status", "hint"),
    [
        (401, "API key was rejected"),
        (404, "llm available"),
        (503, "overloaded"),
        (500, "Try again later"),
    ],
)
def test_explain_api_error_gives_a_hint(status: int, hint: str) -> None:
    """Check that common HTTP errors get a short hint on what to do."""
    client = fake_client(lambda request: httpx2.Response(status, json={"error": {"message": "x"}}))
    with pytest.raises(openai.APIStatusError) as exc_info:
        client.complete([{"role": "user", "content": "Hello"}])
    assert hint in explain_api_error(exc_info.value)


def test_explain_api_error_reads_google_style_errors() -> None:
    """Check that the provider's message is shown when the error comes wrapped in a list."""
    body = [{"error": {"code": 404, "message": "This model is no longer available."}}]
    client = fake_client(lambda request: httpx2.Response(404, json=body))
    with pytest.raises(openai.NotFoundError) as exc_info:
        client.complete([{"role": "user", "content": "Hello"}])
    assert "no longer available" in explain_api_error(exc_info.value)


def test_timeout_is_reported_as_no_answer() -> None:
    """Check that a timeout says the provider didn't answer, not that the internet is down."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    with pytest.raises(openai.APITimeoutError) as exc_info:
        fake_client(handler).complete([{"role": "user", "content": "Hello"}])
    assert "didn't answer in time" in explain_api_error(exc_info.value)
