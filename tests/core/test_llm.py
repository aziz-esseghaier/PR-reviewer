"""Tests for the LLM client in core/llm.py.

Checks what the client sends to a provider, how it reads replies and errors, and how it uses
the response cache. Every test runs against a fake provider, so none uses API quota or keys.

How to run (from the project folder):
    pytest tests/core/test_llm.py
"""

import json

import httpx2
import openai
import pytest

from pr_reviewer.core.config import ConfigError
from pr_reviewer.core.llm import ChatMessage, LLMClient, explain_api_error, list_provider_models
from tests.fakes import FAKE_CONFIG, FakeProvider, MemoryCache, fake_client


def test_complete_sends_profile_settings_and_reads_reply() -> None:
    """Check the request carries the profile's model and settings, and the reply is parsed."""
    provider = FakeProvider("Hi there.")
    response = fake_client(provider).complete(
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}]
    )

    request = provider.requests[0]
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
    assert response.cached is False


def test_empty_reply_becomes_empty_text() -> None:
    """Check that a reply cut off before any text gives "" and finish reason "length"."""
    client = fake_client(FakeProvider(None, "length"))
    response = client.complete([{"role": "user", "content": "Hello"}])
    assert response.text == ""
    assert response.finish_reason == "length"


def test_repeated_request_is_answered_from_cache() -> None:
    """Check that asking the same thing twice sends one request and marks the second as cached."""
    provider = FakeProvider("Hi there.")
    client = fake_client(provider, cache=MemoryCache())
    messages: list[ChatMessage] = [{"role": "user", "content": "Hello"}]

    first = client.complete(messages)
    second = client.complete(messages)

    assert len(provider.requests) == 1
    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text


def test_changed_prompt_or_sample_asks_again() -> None:
    """Check that a different prompt, or a new sample number, goes to the provider."""
    provider = FakeProvider("Hi there.")
    client = fake_client(provider, cache=MemoryCache())

    client.complete([{"role": "user", "content": "Hello"}])
    client.complete([{"role": "user", "content": "Hello!"}])
    client.complete([{"role": "user", "content": "Hello"}], sample=1)
    client.complete([{"role": "user", "content": "Hello"}], sample=1)

    assert len(provider.requests) == 3


def test_empty_reply_is_not_cached() -> None:
    """Check that an empty reply isn't saved, so the next call asks the provider again."""
    provider = FakeProvider(None, "length")
    cache = MemoryCache()
    client = fake_client(provider, cache=cache)

    client.complete([{"role": "user", "content": "Hello"}])
    client.complete([{"role": "user", "content": "Hello"}])

    assert len(provider.requests) == 2
    assert cache.entries == {}


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
        LLMClient("fake-small", FAKE_CONFIG)


def test_unknown_profile_is_rejected() -> None:
    """Check that asking for a profile that isn't configured fails before any request."""
    with pytest.raises(ConfigError, match="No model profile"):
        LLMClient("missing", FAKE_CONFIG, api_key="test-key")


def test_list_provider_models_returns_sorted_ids() -> None:
    """Check that the provider's model list comes back as sorted IDs."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/models"
        models = [{"id": name, "object": "model", "created": 0, "owned_by": "x"} for name in "ba"]
        return httpx2.Response(200, json={"object": "list", "data": models})

    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    provider = FAKE_CONFIG.provider("fake")
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
