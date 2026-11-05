"""Test doubles shared by the test files: fake LLM providers, a fake clock, in-memory versions
of the response cache and the usage log, and a LangSmith client that keeps traces in memory.

The fake providers answer OpenAI-style requests without touching the network, so tests use no
API quota and need no keys. FakeProvider gives one reply and can fail a set number of times
first, to test retries. ScriptedProvider plays a list of different replies and errors, one per
request, to test multi-step graphs.
The fake clock only moves when something sleeps on it, so tests of waits and retries take no
real time. The in-memory cache and usage log stand in for their PostgreSQL versions in tests
that don't need a database. FakeLangSmith catches the traced steps so tests can check them
without uploading anything. Test files import from here; the file isn't run directly.
"""

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2
import langsmith

from pr_reviewer.core.cache import request_key
from pr_reviewer.core.config import ModelLimits, ModelProfile, ModelsConfig, ProviderConfig
from pr_reviewer.core.limits import Clock
from pr_reviewer.core.llm import LLMClient, LLMRequest, LLMResponse, ResponseCache
from pr_reviewer.core.usage import UsageEntry, UsageLog

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
        time (float): Seconds since the clock started.
        start (datetime): The date and time the clock started at.
        sleeps (list[float]): Every wait asked for, in order.
    """

    def __init__(self, start: datetime = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)) -> None:
        """Start the clock.

        Args:
            start (datetime): The date and time to start at.
        """
        self.time = 0.0
        self.start = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        """Return the fake time.

        Returns:
            float: Seconds since the clock started.
        """
        return self.time

    def wall_time(self) -> datetime:
        """Return the fake date and time.

        Returns:
            datetime: The start time plus the seconds slept so far.
        """
        return self.start + timedelta(seconds=self.time)

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


class ScriptedProvider:
    """A pretend provider that answers each request with the next step of a script.

    Attributes:
        requests (list[httpx2.Request]): Every request received, oldest first.
        script (list[str | int | httpx2.Response]): Steps still to play, in order. Text is a
            reply; a number is an HTTP error code to answer with; a response is sent as it is,
            such as a reply cut off at the length limit.
    """

    def __init__(self, script: Sequence[str | int | httpx2.Response]) -> None:
        """Set the replies to give.

        Args:
            script (Sequence[str | int | httpx2.Response]): One step per expected request,
                such as ['{"verdict": "upheld"}', 401].
        """
        self.requests: list[httpx2.Request] = []
        self.script = list(script)

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        """Record a request and answer it with the next step.

        Args:
            request (httpx2.Request): The request the SDK sent.

        Returns:
            httpx2.Response: The next reply or error. With the script used up, a 400 error,
            which the client doesn't retry, so the test fails clearly.
        """
        self.requests.append(request)
        step = self.script.pop(0) if self.script else 400
        if isinstance(step, httpx2.Response):
            return step
        if isinstance(step, int):
            return httpx2.Response(step, json={"error": {"message": f"fake error {step}"}})
        return httpx2.Response(200, json=completion_body(step))

    def body(self, index: int) -> dict[str, Any]:
        """Read the JSON body of one received request.

        Args:
            index (int): Which request, 0 for the first.

        Returns:
            dict[str, Any]: The body, with the model, messages and settings.
        """
        body: dict[str, Any] = json.loads(self.requests[index].content)
        return body

    def messages(self, index: int) -> list[dict[str, str]]:
        """Read the messages of one received request.

        Args:
            index (int): Which request, 0 for the first.

        Returns:
            list[dict[str, str]]: Its messages, each with "role" and "content".
        """
        body: dict[str, Any] = json.loads(self.requests[index].content)
        messages: list[dict[str, str]] = body["messages"]
        return messages


def config_with(limits: ModelLimits, daily_reset: str | None = None) -> ModelsConfig:
    """Build a config with one fake provider and one profile, "fake-custom", with given limits.

    Args:
        limits (ModelLimits): The profile's limits.
        daily_reset (str | None): The provider's daily reset rule. None means unknown.

    Returns:
        ModelsConfig: The config.
    """
    return ModelsConfig(
        providers={
            "fake": ProviderConfig(
                base_url="https://llm.example.test/v1",
                api_key_env="FAKE_LLM_KEY",
                daily_reset=daily_reset,
            )
        },
        models={
            "fake-custom": ModelProfile(
                provider="fake", model="fake-model-v1", max_tokens=64, limits=limits
            )
        },
    )


def fake_client(
    handler: Handler,
    cache: ResponseCache | None = None,
    clock: Clock | None = None,
    profile: str = "fake-small",
    usage: UsageLog | None = None,
    config: ModelsConfig = FAKE_CONFIG,
) -> LLMClient:
    """Create a client for a fake profile whose requests go to a handler, not the network.

    Args:
        handler (Handler): Function that receives each request and returns the response,
            such as a FakeProvider.
        cache (ResponseCache | None): Cache for the client. None turns caching off.
        clock (Clock | None): Clock for waits. None uses a new FakeClock, so no test ever
            really sleeps.
        profile (str): Profile name. In FAKE_CONFIG: "fake-small" (no limits) or
            "fake-limited" (2 requests a minute).
        usage (UsageLog | None): Usage log for the client. None turns logging off.
        config (ModelsConfig): Config holding the profile, such as one from config_with().

    Returns:
        LLMClient: The client.
    """
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return LLMClient(
        profile,
        config,
        cache=cache,
        usage=usage,
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


class FakeLangSmith(langsmith.Client):
    """A LangSmith client that keeps traced steps in memory instead of uploading them.

    LangSmith sends each step twice: when it starts and when it ends. Both methods that send
    are replaced here, so nothing reaches the network.

    Attributes:
        caught (dict[str, dict[str, Any]]): Every traced step by its ID, oldest first: what
            was sent at the start, updated with what was sent at the end. (The real client
            already uses the name `runs`.)
    """

    def __init__(self) -> None:
        """Create the client. Its address is never contacted."""
        super().__init__(
            api_key="fake-langsmith-key", api_url="http://127.0.0.1:9", auto_batch_tracing=False
        )
        self.caught: dict[str, dict[str, Any]] = {}

    def create_run(self, name: str, inputs: dict[str, Any], run_type: Any, **kwargs: Any) -> None:
        """Keep a step that just started.

        Args:
            name (str): The step's name.
            inputs (dict[str, Any]): Its inputs.
            run_type (Any): Its type, such as "chain" or "llm".
            **kwargs (Any): Everything else LangSmith sends, such as id and parent_run_id.
        """
        self.caught[str(kwargs["id"])] = {
            **kwargs,
            "name": name,
            "inputs": inputs,
            "run_type": run_type,
        }

    def update_run(self, run_id: Any, **kwargs: Any) -> None:
        """Add what LangSmith sends when a step ends, such as its outputs or error.

        Args:
            run_id (Any): The step's ID.
            **kwargs (Any): The fields sent. Those sent as None are left out.
        """
        sent = {key: value for key, value in kwargs.items() if value is not None}
        self.caught.setdefault(str(run_id), {}).update(sent)

    def steps(self, run_type: str | None = None) -> list[dict[str, Any]]:
        """List the traced steps, oldest first.

        Args:
            run_type (str | None): Only steps of this type, such as "llm". None lists all.

        Returns:
            list[dict[str, Any]]: The steps.
        """
        return [
            run for run in self.caught.values() if run_type is None or run["run_type"] == run_type
        ]

    def named(self, name: str) -> dict[str, Any]:
        """Find the one step with a given name.

        Args:
            name (str): The step's name, such as "verify".

        Returns:
            dict[str, Any]: The step.
        """
        (run,) = [run for run in self.caught.values() if run["name"] == name]
        return run


def metadata_of(run: dict[str, Any]) -> dict[str, Any]:
    """Read the metadata of a step caught by FakeLangSmith.

    Args:
        run (dict[str, Any]): The step.

    Returns:
        dict[str, Any]: Its metadata.
    """
    metadata: dict[str, Any] = run["extra"]["metadata"]
    return metadata


class MemoryUsageLog:
    """A usage log kept in a list, for tests that don't need the database.

    Attributes:
        entries (list[UsageEntry]): Every entry recorded, oldest first.
    """

    def __init__(self, entries: Sequence[UsageEntry] = ()) -> None:
        """Start with some earlier entries, or none.

        Args:
            entries (Sequence[UsageEntry]): Entries to start with, oldest first.
        """
        self.entries = list(entries)

    def record(self, entry: UsageEntry) -> None:
        """Save one entry.

        Args:
            entry (UsageEntry): The request or cache hit to save.
        """
        self.entries.append(entry)

    def recent(self, provider: str, model: str, since: datetime) -> list[tuple[datetime, int]]:
        """List a model's requests since a moment, without cache hits.

        Args:
            provider (str): Provider name.
            model (str): Model ID.
            since (datetime): Only requests after this moment count.

        Returns:
            list[tuple[datetime, int]]: (time, tokens) of each request, oldest first.
        """
        return [
            (entry.time, entry.input_tokens + entry.output_tokens)
            for entry in self.entries
            if entry.provider == provider
            and entry.model == model
            and entry.outcome != "cache_hit"
            and entry.time > since
        ]
