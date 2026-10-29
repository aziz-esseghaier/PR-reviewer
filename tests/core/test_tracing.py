"""Tests for LangSmith tracing of model calls in core/tracing.py.

Runs LLMClient against fake providers with tracing caught in memory (the `traces` fixture), so
nothing is uploaded. Checks what a model-call step holds (messages, reply, tokens, cache flag,
retries, errors), that no API key is traced, that nothing is traced with tracing off, and the
start-up check that turns tracing off when the LangSmith key is missing.

How to run (from the project folder):
    pytest tests/core/test_tracing.py
"""

from typing import Any

import openai
import pytest
from langsmith.run_helpers import tracing_context

from pr_reviewer.core.llm import ChatMessage
from pr_reviewer.core.tracing import describe_tracing, flush_traces, start_tracing, tracing_on
from tests.fakes import FakeLangSmith, FakeProvider, MemoryCache, fake_client, metadata_of

MESSAGES = [ChatMessage(role="user", content="Say hi.")]


def test_model_call_is_traced_with_messages_reply_and_tokens(traces: FakeLangSmith) -> None:
    """Check the step of a call sent to the provider."""
    fake_client(FakeProvider("Hi there.")).complete(MESSAGES)
    (step,) = traces.steps()
    assert step["name"] == "fake-small"
    assert step["run_type"] == "llm"
    assert step["inputs"] == {"messages": MESSAGES}
    assert step["outputs"]["choices"][0]["message"]["content"] == "Hi there."
    assert step["outputs"]["usage_metadata"] == {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }
    metadata = metadata_of(step)
    assert metadata["ls_provider"] == "fake"
    assert metadata["ls_model_name"] == "fake-model-v1"
    assert metadata["cached"] is False
    assert metadata["attempts"] == 1


def test_cache_hit_is_traced_without_tokens(traces: FakeLangSmith) -> None:
    """Check that a reply from the cache shows as cached and reports no tokens used."""
    client = fake_client(FakeProvider("Hi there."), cache=MemoryCache())
    client.complete(MESSAGES)
    client.complete(MESSAGES)
    first, second = traces.steps("llm")
    assert metadata_of(first)["cached"] is False
    assert metadata_of(second)["cached"] is True
    assert "usage_metadata" not in second["outputs"]
    assert second["outputs"]["choices"][0]["message"]["content"] == "Hi there."


def test_retries_stay_in_one_step(traces: FakeLangSmith) -> None:
    """Check that a call that needed a retry is one step that counts its tries."""
    fake_client(FakeProvider(failures=[503])).complete(MESSAGES)
    (step,) = traces.steps()
    assert metadata_of(step)["attempts"] == 2


def test_failed_call_is_traced_with_its_error(traces: FakeLangSmith) -> None:
    """Check that a call that raises still shows, with the error."""
    with pytest.raises(openai.AuthenticationError):
        fake_client(FakeProvider(failures=[401])).complete(MESSAGES)
    (step,) = traces.steps()
    assert "AuthenticationError" in step["error"]
    assert not step.get("outputs")


def test_api_key_is_never_traced(traces: FakeLangSmith) -> None:
    """Check that the provider key appears nowhere in what is sent to LangSmith."""
    fake_client(FakeProvider()).complete(MESSAGES)
    assert traces.caught
    assert "test-key" not in repr(traces.caught)


def test_nothing_is_traced_when_tracing_is_off() -> None:
    """Check that with tracing off no step reaches the LangSmith client."""
    fake = FakeLangSmith()
    with tracing_context(client=fake):
        fake_client(FakeProvider()).complete(MESSAGES)
    assert fake.caught == {}
    assert describe_tracing() == "off"


def test_tracing_without_api_key_is_turned_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Check that tracing asked for without a LangSmith key is turned off with a warning."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    with tracing_context(enabled=True):
        start_tracing()
        assert not tracing_on()
    assert "LANGSMITH_API_KEY is empty" in caplog.text


def test_tracing_with_api_key_stays_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that tracing with a key set stays on and names its project."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-placeholder")
    with tracing_context(enabled=True):
        start_tracing()
        assert tracing_on()
        assert describe_tracing().startswith("on (LangSmith project ")


def test_failed_upload_never_fails_the_command(
    traces: FakeLangSmith, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Check that an error while uploading the last traces is only a warning."""

    def broken_flush(*args: Any, **kwargs: Any) -> None:
        raise ConnectionError("LangSmith is down")

    monkeypatch.setattr(traces, "flush", broken_flush)
    flush_traces()
    assert "LangSmith is down" in caplog.text
