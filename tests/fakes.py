"""Test doubles shared by the test files: a fake LLM provider, a fake clock and an in-memory cache.

The fake provider answers OpenAI-style requests without touching the network, so tests use no
API quota and need no keys. It can also fail a set number of times first, to test retries.
The fake clock only moves when something sleeps on it, so tests of waits and retries take no
real time. The in-memory cache stands in for the PostgreSQL cache in tests that don't need a
database. Test files import from here; the file isn't run directly.
"""

from collections.abc import Callable, Sequence
from typing import Any

import httpx2

from pr_reviewer.core.cache import request_key
from pr_reviewer.core.config import ModelLimits, ModelProfile, ModelsConfig, ProviderConfig
from pr_reviewer.core.limits import Clock
from pr_reviewer.core.llm import LLMClient, LLMRequest, LLMResponse, ResponseCache

Handler = Callable[[httpx2.Request], httpx2.Response]

# One fake provider with two profiles: one without limits and one allowing 2 requests a minute.
# The key variable is never set, so tests that need a client pass api_key directly.
FAKE_CONFIG = ModelsConfig(
    providers={
        "fake": ProviderConfig(base_url="https://llm.example.test/v1", api_key_env="FAKE_LLM_KEY")
    },
    models={
        "fake-small": ModelProfile(
            provider="fake", model="fake-model-v1", temperature=0.0, max_tokens=64
        ),
        "fake-limited": ModelProfile(
            provider="fake",
            model="fake-model-v1",
            temperature=0.0,
            max_tokens=64,
            limits=ModelLimits(requests_per_minute=2),
        ),
    },
)


class FakeClock:
    """A clock that stands still until something sleeps on it, so waits take no real time.

    Attributes:
        time (float): The current fake time in seconds.
        sleeps (list[float]): Every wait asked for, in order.
    """

    def __init__(self) -> None:
        """Start the clock at 0."""
        self.time = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        """Return the fake time.

        Returns:
            float: Seconds since the clock started.
        """
        return self.time

    def sleep(self, seconds: float) -> None:
        """Record a wait and move the clock forward by it, at once.

        Args:
            seconds (float): How long the caller wanted to wait.
        """
        self.sleeps.append(seconds)
        self.time += seconds


def completion_body(content: str | None, finish_reason: str = "stop") -> dict[str, Any]:
    """Build a chat completion reply the way OpenAI-compatible providers send it.

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


class FakeProvider:
    """A pretend provider that records every request, can fail a few times, then answers.

    Attributes:
        requests (list[httpx2.Request]): Every request received, oldest first.
        reply (str | None): Text to answer with. None means an empty reply.
        finish_reason (str): Finish reason to report.
        failures (list[int]): HTTP error codes to answer the first requests with, in order.
    """

    def __init__(
        self,
        reply: str | None = "Hi there.",
        finish_reason: str = "stop",
        failures: Sequence[int] = (),
    ) -> None:
        """Set how the provider will answer.

        Args:
            reply (str | None): Text to answer with. None means an empty reply.
            finish_reason (str): Finish reason to report.
            failures (Sequence[int]): HTTP error codes for the first requests, such as
                (503, 503) to fail twice before answering.
        """
        self.requests: list[httpx2.Request] = []
        self.reply = reply
        self.finish_reason = finish_reason
        self.failures = list(failures)

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        """Record a request and answer it, with an error while failures remain.

        Args:
            request (httpx2.Request): The request the SDK sent.

        Returns:
            httpx2.Response: The next error, or a 200 reply with the configured text.
        """
        self.requests.append(request)
        if self.failures:
            status = self.failures.pop(0)
            return httpx2.Response(status, json={"error": {"message": f"fake error {status}"}})
        return httpx2.Response(200, json=completion_body(self.reply, self.finish_reason))


def fake_client(
    handler: Handler,
    cache: ResponseCache | None = None,
    clock: Clock | None = None,
    profile: str = "fake-small",
) -> LLMClient:
    """Create a client for a fake profile whose requests go to a handler, not the network.

    Args:
        handler (Handler): Function that receives each request and returns the response,
            such as a FakeProvider.
        cache (ResponseCache | None): Cache for the client. None turns caching off.
        clock (Clock | None): Clock for waits. None uses a new FakeClock, so no test ever
            really sleeps.
        profile (str): "fake-small" (no limits) or "fake-limited" (2 requests a minute).

    Returns:
        LLMClient: The client.
    """
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return LLMClient(
        profile,
        FAKE_CONFIG,
        cache=cache,
        clock=clock or FakeClock(),
        api_key="test-key",
        http_client=http,
    )


class MemoryCache:
    """A response cache kept in a dict, for tests that don't need the database.

    Attributes:
        entries (dict[str, LLMResponse]): Saved replies by request fingerprint.
    """

    def __init__(self) -> None:
        """Start with an empty cache."""
        self.entries: dict[str, LLMResponse] = {}

    def get(self, request: LLMRequest) -> LLMResponse | None:
        """Look up the saved reply to a request.

        Args:
            request (LLMRequest): The request to look up.

        Returns:
            LLMResponse | None: The saved reply, or None on a miss.
        """
        return self.entries.get(request_key(request))

    def put(self, request: LLMRequest, response: LLMResponse) -> None:
        """Save the reply to a request.

        Args:
            request (LLMRequest): The request that was sent.
            response (LLMResponse): The provider's reply.
        """
        self.entries[request_key(request)] = response
