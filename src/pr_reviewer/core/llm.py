"""LLM client: the single route from our code to a language model.

`LLMClient` is tied to one model profile from configs/models.yaml and sends chat requests to it
through the OpenAI SDK. Groq and Google AI Studio both accept the OpenAI request format, so one
SDK pointed at a different base URL covers both.

For each request the client:
1. answers from the response cache if it has seen the request before (core/cache.py);
2. otherwise checks the model's daily limits against the usage log, and stops if the quota is
   used up (core/limits.py, core/usage.py);
3. waits, if needed, to stay under the model's per-minute limits (core/limits.py);
4. sends it, retrying after rate limits, overload and timeouts (core/limits.py);
5. records every try in the usage log, and saves the reply in the cache.
Cache hits are recorded in the usage log too, but never count against a limit. When tracing is
on, each call, cached or not, is also a step in LangSmith (core/tracing.py).

The file also lists a provider's models and turns provider errors into short hints.
Other modules import it; the file isn't run directly.
"""

import logging
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal, Protocol, TypedDict, cast

import httpx2
import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    completion_create_params,
)
from openai.types.shared import ReasoningEffort
from pydantic import BaseModel, ConfigDict

from pr_reviewer.core.config import ModelsConfig, ProviderConfig, describe_reset, read_api_key
from pr_reviewer.core.limits import (
    DAY,
    Clock,
    DailyStatus,
    LimitError,
    RetryPolicy,
    SystemClock,
    Throttle,
    call_with_retries,
    daily_status,
    describe_wait,
    estimate_tokens,
)
from pr_reviewer.core.tracing import trace_model_call
from pr_reviewer.core.usage import Outcome, UsageEntry, UsageLog, outcome_of

log = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    """One message in a conversation with the model.

    Attributes:
        role (str): "system" for instructions, "user" for input, "assistant" for model replies.
        content (str): The message text.
    """

    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    """Everything that decides a model's answer. The cache uses it as the lookup key.

    The profile name isn't part of it, so renaming a profile keeps its cached replies.

    Attributes:
        provider (str): Provider name, such as "groq".
        model (str): Model ID as the provider spells it.
        messages (list[ChatMessage]): The conversation, oldest first.
        temperature (float): Sampling randomness.
        max_tokens (int): Longest reply allowed, in tokens.
        sample (int): Which answer to this exact request. 0 is the normal one. 1, 2, ... ask the
            provider again and are cached separately.
        response_format (dict[str, Any] | None): The reply format to ask for, in the OpenAI
            format, such as {"type": "json_schema", "json_schema": {...}}. None means plain
            text. Left at None, it isn't part of the cache key, so older keys stay valid.
        reasoning_effort (str | None): How long a reasoning model thinks, such as "medium".
            Like response_format, left out of the cache key when None.
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    messages: list[ChatMessage]
    temperature: float
    max_tokens: int
    sample: int = 0
    response_format: dict[str, Any] | None = None
    reasoning_effort: str | None = None


class LLMResponse(BaseModel):
    """A model's reply, plus what the call cost and how long it took.

    Attributes:
        profile (str): Name of the model profile that answered.
        provider (str): Name of the provider, such as "groq".
        model (str): Model ID the provider reported.
        text (str): The reply. Empty if the model wrote nothing.
        finish_reason (str): Why the model stopped: "stop" means done, "length" means it hit
            max_tokens.
        input_tokens (int): Tokens in the prompt.
        output_tokens (int): Tokens in the reply.
        latency_s (float): Seconds from sending the request to receiving the reply.
        attempts (int): Requests it took, 1 if the first one worked. Retries add to it.
        waited_s (float): Seconds spent waiting before the reply: for the per-minute limits
            and between retries.
        cached (bool): True if the reply came from the cache. Tokens, timing, attempts and
            waits are then those of the original call, and no request was sent this time.
    """

    model_config = ConfigDict(frozen=True)

    profile: str
    provider: str
    model: str
    text: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    attempts: int = 1
    waited_s: float = 0.0
    cached: bool = False


class ResponseCache(Protocol):
    """Anything that can save replies and look them up by request.

    LLMClient only needs these two methods, so real runs pass the PostgreSQL cache
    (core/cache.py) and tests pass a simple in-memory one.
    """

    def get(self, request: LLMRequest) -> LLMResponse | None:
        """Look up the saved reply to a request.

        Args:
            request (LLMRequest): The request to look up.

        Returns:
            LLMResponse | None: The saved reply, or None if there isn't one.
        """
        ...

    def put(self, request: LLMRequest, response: LLMResponse) -> None:
        """Save the reply to a request.

        Args:
            request (LLMRequest): The request that was sent.
            response (LLMResponse): The provider's reply.
        """
        ...


class LLMError(Exception):
    """Raised when a provider's reply can't be used."""


def _make_sdk(
    provider: ProviderConfig, api_key: str | None, http_client: httpx2.Client | None
) -> openai.OpenAI:
    """Create an OpenAI SDK client aimed at a provider's endpoint.

    Args:
        provider (ProviderConfig): The provider to talk to.
        api_key (str | None): Key to use. None reads it from the provider's environment variable.
        http_client (httpx2.Client | None): HTTP client to send requests with. Tests pass one
            with a fake transport. None uses the SDK's default.

    Returns:
        openai.OpenAI: The SDK client.
    """
    return openai.OpenAI(
        api_key=api_key or read_api_key(provider),
        base_url=provider.base_url,
        timeout=provider.timeout_s,
        # The SDK would otherwise retry rate-limit errors on its own, out of our sight.
        # core/limits.py retries instead, so every request is counted and logged.
        max_retries=0,
        http_client=http_client,
    )


def _to_sdk_message(message: ChatMessage) -> ChatCompletionMessageParam:
    """Convert one of our messages into the SDK's message type.

    Args:
        message (ChatMessage): The message to convert.

    Returns:
        ChatCompletionMessageParam: The same message in the SDK's format.
    """
    match message["role"]:
        case "system":
            return ChatCompletionSystemMessageParam(role="system", content=message["content"])
        case "user":
            return ChatCompletionUserMessageParam(role="user", content=message["content"])
        case "assistant":
            return ChatCompletionAssistantMessageParam(role="assistant", content=message["content"])


class LLMClient:
    """Sends chat requests to one model profile, answering from the cache when it can.

    A client is tied to one model, so a run that uses one client can't mix models.

    Attributes:
        name (str): Profile name, such as "groq-gpt-oss-120b".
        profile (ModelProfile): The model, its sampling settings and its limits.
        provider_name (str): Name of the provider, such as "google".
    """

    def __init__(
        self,
        name: str,
        config: ModelsConfig,
        *,
        cache: ResponseCache | None = None,
        usage: UsageLog | None = None,
        retry_policy: RetryPolicy | None = None,
        clock: Clock | None = None,
        api_key: str | None = None,
        http_client: httpx2.Client | None = None,
    ) -> None:
        """Set up a client for one model profile.

        Args:
            name (str): Profile name from configs/models.yaml.
            config (ModelsConfig): The loaded model config.
            cache (ResponseCache | None): Where to look up and save replies. None turns
                caching off, so every call goes to the provider.
            usage (UsageLog | None): Where to record requests and check daily limits. None
                turns both off.
            retry_policy (RetryPolicy | None): How many tries and how long to wait after
                temporary errors. None uses the defaults.
            clock (Clock | None): Used to wait. Tests pass a fake one. None uses the real clock.
            api_key (str | None): Key to use. None reads it from the provider's environment
                variable.
            http_client (httpx2.Client | None): HTTP client for the SDK. Tests pass one with a
                fake transport. None uses the SDK's default.

        Raises:
            ConfigError: If the profile doesn't exist or its API key isn't set.
        """
        self.name = name
        self.profile = config.profile(name)
        self.provider_name = self.profile.provider
        self._reset_rule = config.provider(self.provider_name).reset_rule
        self._cache = cache
        self._usage = usage
        self._retry_policy = retry_policy or RetryPolicy()
        self._clock = clock or SystemClock()
        self._throttle = Throttle(
            self.profile.limits.requests_per_minute,
            self.profile.limits.tokens_per_minute,
            clock=self._clock,
            label=name,
        )
        self._sdk = _make_sdk(config.provider(self.provider_name), api_key, http_client)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        sample: int = 0,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Get the model's reply to a conversation, from the cache if it was asked before.

        Args:
            messages (Sequence[ChatMessage]): The conversation so far, oldest first.
            sample (int): Which answer to this exact request. Keep 0 normally. Pass 1, 2, ...
                to ask the provider again, for example when a cached reply was unusable.
                Each sample is cached separately. At temperature 0 a new sample is often
                the same text, because the model has little randomness to draw on.
            response_format (dict[str, Any] | None): Ask for replies in a fixed format, such
                as a JSON schema, in the OpenAI format. The provider must support it. None
                asks for plain text. The reply text still needs checking: not every
                provider enforces the format strictly.

        Returns:
            LLMResponse: The reply text, token counts, timing and whether it came from the cache.

        Raises:
            openai.APIError: If the provider rejects the request, or still fails after all
                retries.
            LimitError: If waiting won't help: a daily quota is used up, or the request is
                bigger than a limit could ever allow.
            LLMError: If the provider answers without any reply.
        """
        request = LLMRequest(
            provider=self.provider_name,
            model=self.profile.model,
            messages=list(messages),
            temperature=self.profile.temperature,
            max_tokens=self.profile.max_tokens,
            sample=sample,
            response_format=response_format,
            reasoning_effort=self.profile.reasoning_effort,
        )
        with trace_model_call(self.name, request) as trace:
            response = self._answer(request)
            trace.record(response)
        return response

    def _answer(self, request: LLMRequest) -> LLMResponse:
        """Answer a request from the cache, or send it and save the reply in the cache.

        Args:
            request (LLMRequest): The request to answer.

        Returns:
            LLMResponse: The reply.
        """
        if self._cache is not None:
            saved = self._cache.get(request)
            if saved is not None:
                self._record(
                    "cache_hit",
                    sent_at=self._clock.wall_time(),
                    input_tokens=saved.input_tokens,
                    output_tokens=saved.output_tokens,
                )
                return saved.model_copy(update={"cached": True, "profile": self.name})
        response = self._send(request)
        # An empty reply is usually a glitch, so we don't keep it: the next call asks again.
        if self._cache is not None and response.text:
            self._cache.put(request, response)
        return response

    def _send(self, request: LLMRequest) -> LLMResponse:
        """Send a request to the provider, within the limits and with retries, and read the reply.

        Args:
            request (LLMRequest): The model, messages and settings to send.

        Returns:
            LLMResponse: The provider's reply, with cached set to False.

        Raises:
            openai.APIError: If the provider rejects the request or still fails after all
                retries.
            LimitError: If waiting won't help.
            LLMError: If the provider answers without any reply.
        """
        estimate = estimate_tokens(
            [message["content"] for message in request.messages], request.max_tokens
        )
        limit_wait_s = 0.0

        def one_try() -> tuple[ChatCompletion, float]:
            """Check the daily limits, wait for room per minute, then send the request once.

            Every try is recorded in the usage log, whether it worked or not.

            Returns:
                tuple[ChatCompletion, float]: The SDK's reply and its latency in seconds.
            """
            nonlocal limit_wait_s
            limit_wait_s += self._wait_for_daily_room(estimate)
            slot, waited_s = self._throttle.reserve(estimate)
            limit_wait_s += waited_s
            sent_at = self._clock.wall_time()
            started = time.perf_counter()
            reply_format = (
                openai.omit
                if request.response_format is None
                else cast(completion_create_params.ResponseFormat, request.response_format)
            )
            effort = (
                openai.omit
                if request.reasoning_effort is None
                else cast(ReasoningEffort, request.reasoning_effort)
            )
            try:
                completion = self._sdk.chat.completions.create(
                    model=request.model,
                    messages=[_to_sdk_message(message) for message in request.messages],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    response_format=reply_format,
                    reasoning_effort=effort,
                )
            except openai.APIError as exc:
                outcome, status = outcome_of(exc)
                self._record(outcome, sent_at, status, latency_s=time.perf_counter() - started)
                raise
            latency_s = time.perf_counter() - started
            usage = completion.usage
            if usage is not None:
                slot.tokens = usage.total_tokens  # replace the estimate
            self._record(
                "ok",
                sent_at,
                200,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_s=latency_s,
            )
            return completion, latency_s

        result = call_with_retries(
            one_try, label=self.name, policy=self._retry_policy, clock=self._clock
        )
        completion, latency_s = result.value
        if not completion.choices:
            raise LLMError(f"{self.name}: the provider returned no reply")
        choice = completion.choices[0]
        usage = completion.usage
        return LLMResponse(
            profile=self.name,
            provider=self.provider_name,
            model=completion.model or request.model,
            text=choice.message.content or "",
            finish_reason=choice.finish_reason or "unknown",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_s=round(latency_s, 3),
            attempts=result.attempts,
            waited_s=round(result.waited_s + limit_wait_s, 3),
        )

    def _wait_for_daily_room(self, estimate: int) -> float:
        """Make sure the next request fits under the model's daily limits.

        Reads the model's requests of the last 24 hours from the usage log and applies the
        provider's reset rule. A short wait (a quota that refills gradually, or a reset due in
        a moment) is waited out. A long one stops the run with a message saying when the
        quota comes back.

        Args:
            estimate (int): Estimated tokens of the request.

        Returns:
            float: Seconds spent waiting. 0 if there was room.

        Raises:
            LimitError: If a daily quota is used up for longer than RetryPolicy.max_wait_s,
                or the request alone is bigger than the daily token limit.
        """
        limits = self.profile.limits
        if self._usage is None or (
            limits.requests_per_day is None and limits.tokens_per_day is None
        ):
            return 0.0
        if limits.tokens_per_day is not None and estimate > limits.tokens_per_day:
            raise LimitError(
                f"{self.name}: this request needs about {estimate} tokens, more than the "
                f"model's {limits.tokens_per_day} tokens per day."
            )
        waited_s = 0.0
        while True:
            now = self._clock.wall_time()
            history = self._usage.recent(self.provider_name, self.profile.model, now - DAY)
            checks: list[tuple[str, DailyStatus]] = []
            if limits.requests_per_day is not None:
                requests = [(when, 1) for when, _ in history]
                status = daily_status(requests, limits.requests_per_day, self._reset_rule, now, 1)
                checks.append(("requests", status))
            if limits.tokens_per_day is not None:
                status = daily_status(
                    history, limits.tokens_per_day, self._reset_rule, now, estimate
                )
                checks.append(("tokens", status))
            kind, worst = max(checks, key=lambda check: check[1].wait_s)
            if worst.wait_s <= 0:
                return waited_s
            if worst.wait_s > self._retry_policy.max_wait_s:
                raise LimitError(
                    f"{self.name}: daily {kind} limit reached ({worst.used} of {worst.limit} "
                    f"{kind}; {describe_reset(self._reset_rule)}). "
                    f"Room again {describe_wait(worst.wait_s, now)}."
                )
            log.info("%s: waiting %.1f s for the daily %s limit", self.name, worst.wait_s, kind)
            self._clock.sleep(worst.wait_s)
            waited_s += worst.wait_s

    def _record(
        self,
        outcome: Outcome,
        sent_at: datetime,
        status_code: int | None = None,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_s: float | None = None,
    ) -> None:
        """Write one entry to the usage log, if there is one.

        Args:
            outcome (Outcome): What happened: "ok", "cache_hit" or the kind of failure.
            sent_at (datetime): When the request was sent, or the cache was asked.
            status_code (int | None): HTTP status, if there was one.
            input_tokens (int): Prompt tokens.
            output_tokens (int): Reply tokens.
            latency_s (float | None): Seconds the request took. None for cache hits.
        """
        if self._usage is None:
            return
        self._usage.record(
            UsageEntry(
                time=sent_at,
                provider=self.provider_name,
                model=self.profile.model,
                profile=self.name,
                outcome=outcome,
                status_code=status_code,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=None if latency_s is None else round(latency_s, 3),
            )
        )


def list_provider_models(
    provider: ProviderConfig,
    *,
    api_key: str | None = None,
    http_client: httpx2.Client | None = None,
) -> list[str]:
    """Ask a provider which models the API key can use.

    Args:
        provider (ProviderConfig): The provider to ask.
        api_key (str | None): Key to use. None reads it from the provider's environment variable.
        http_client (httpx2.Client | None): HTTP client for the SDK. None uses the SDK's default.

    Returns:
        list[str]: Model IDs, sorted.

    Raises:
        openai.APIError: If the provider rejects the request or can't be reached.
    """
    sdk = _make_sdk(provider, api_key, http_client)
    return sorted(model.id for model in sdk.models.list())


def _provider_message(exc: openai.APIStatusError) -> str:
    """Pull the provider's own error text out of an error reply.

    The SDK finds the text when the reply looks like OpenAI's ({"error": {"message": ...}}).
    Google wraps that object in a list, so the SDK only shows "Error code: 404".

    Args:
        exc (openai.APIStatusError): The error the SDK raised.

    Returns:
        str: The provider's message, or the SDK's own message if there is none.
    """
    body = exc.body
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        error = body.get("error", body)
        message = error.get("message") if isinstance(error, dict) else None
        if isinstance(message, str):
            return message
    return exc.message


def explain_api_error(exc: openai.APIError) -> str:
    """Describe a provider error in one line, with a hint on what to do.

    Args:
        exc (openai.APIError): The error the SDK raised.

    Returns:
        str: A short description and hint.
    """
    if isinstance(exc, openai.APITimeoutError):
        return (
            "The provider didn't answer in time. The model may be overloaded: "
            "try again later, pick another model, or raise timeout_s in the config."
        )
    if isinstance(exc, openai.APIConnectionError):
        return f"Can't reach the provider ({exc}). Check your internet connection."
    if not isinstance(exc, openai.APIStatusError):
        return f"The provider sent a reply we couldn't read: {exc}"
    hints = {
        400: "Check the request settings. Google also answers 400 when the API key is wrong.",
        401: "The API key was rejected. Check it in .env.",
        403: "The API key isn't allowed to use this model.",
        404: "Unknown or retired model. List the models with: pr-reviewer llm available <provider>",
        429: "Rate limit or daily quota reached. Wait a bit, then try again.",
        503: "The model is overloaded. Try again in a minute, or pick another model.",
    }
    hint = hints.get(exc.status_code, "The provider had a problem. Try again later.")
    message = _provider_message(exc)
    return f"The provider refused the request (HTTP {exc.status_code}): {message}\n{hint}"
