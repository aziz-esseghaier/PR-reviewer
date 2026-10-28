"""Test doubles shared by the test files: a fake LLM provider and an in-memory cache.

The fake provider answers OpenAI-style requests without touching the network, so tests use no
API quota and need no keys. The in-memory cache stands in for the PostgreSQL cache in tests
that don't need a database. Test files import from here; the file isn't run directly.
"""

from collections.abc import Callable
from typing import Any

import httpx2

from pr_reviewer.core.cache import request_key
from pr_reviewer.core.config import ModelProfile, ModelsConfig, ProviderConfig
from pr_reviewer.core.llm import LLMClient, LLMRequest, LLMResponse, ResponseCache

Handler = Callable[[httpx2.Request], httpx2.Response]

# One fake provider with one model profile. Its key variable is never set, so tests that need
# a client pass api_key directly.
FAKE_CONFIG = ModelsConfig(
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
    """A pretend provider that records every request and answers with a fixed reply.

    Attributes:
        requests (list[httpx2.Request]): Every request received, oldest first.
        reply (str | None): Text to answer with. None means an empty reply.
        finish_reason (str): Finish reason to report.
    """

    def __init__(self, reply: str | None = "Hi there.", finish_reason: str = "stop") -> None:
        """Set the reply the provider will give.

        Args:
            reply (str | None): Text to answer with. None means an empty reply.
            finish_reason (str): Finish reason to report.
        """
        self.requests: list[httpx2.Request] = []
        self.reply = reply
        self.finish_reason = finish_reason

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        """Record a request and answer it.

        Args:
            request (httpx2.Request): The request the SDK sent.

        Returns:
            httpx2.Response: A 200 reply with the configured text.
        """
        self.requests.append(request)
        return httpx2.Response(200, json=completion_body(self.reply, self.finish_reason))


def fake_client(handler: Handler, cache: ResponseCache | None = None) -> LLMClient:
    """Create a client for the fake profile whose requests go to a handler, not the network.

    Args:
        handler (Handler): Function that receives each request and returns the response,
            such as a FakeProvider.
        cache (ResponseCache | None): Cache for the client. None turns caching off.

    Returns:
        LLMClient: The client.
    """
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return LLMClient("fake-small", FAKE_CONFIG, cache=cache, api_key="test-key", http_client=http)


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
