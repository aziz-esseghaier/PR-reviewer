"""Send chat requests to one configured model through the OpenAI SDK.

Groq and Google AI Studio both accept the OpenAI request format, so one SDK pointed at a
different base URL covers both. Every model call in the project goes through
`LLMClient.complete`, which is where the cache, rate limits and usage log plug in later.
"""

import time
from collections.abc import Sequence
from typing import Literal, TypedDict

import httpx2
import openai
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ConfigDict

from pr_reviewer.core.config import ModelsConfig, ProviderConfig, read_api_key


class ChatMessage(TypedDict):
    """One message in a conversation with the model.

    Attributes:
        role (str): "system" for instructions, "user" for input, "assistant" for model replies.
        content (str): The message text.
    """

    role: Literal["system", "user", "assistant"]
    content: str


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
        # We handle retries ourselves, so every request is counted and logged.
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
    """Sends chat requests to one model profile.

    A client is tied to one model, so a run that uses one client can't mix models.

    Attributes:
        name (str): Profile name, such as "gemini-2.5-flash".
        profile (ModelProfile): The model and its sampling settings.
        provider_name (str): Name of the provider, such as "google".
    """

    def __init__(
        self,
        name: str,
        config: ModelsConfig,
        *,
        api_key: str | None = None,
        http_client: httpx2.Client | None = None,
    ) -> None:
        """Set up a client for one model profile.

        Args:
            name (str): Profile name from configs/models.yaml.
            config (ModelsConfig): The loaded model config.
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
        self._sdk = _make_sdk(config.provider(self.provider_name), api_key, http_client)

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        """Send a conversation to the model and return its reply.

        Args:
            messages (Sequence[ChatMessage]): The conversation so far, oldest first.

        Returns:
            LLMResponse: The reply text, token counts and timing.

        Raises:
            openai.APIError: If the provider rejects the request or can't be reached.
                openai.RateLimitError is the one to watch on free tiers.
            LLMError: If the provider answers without any reply.
        """
        started = time.perf_counter()
        completion = self._sdk.chat.completions.create(
            model=self.profile.model,
            messages=[_to_sdk_message(message) for message in messages],
            temperature=self.profile.temperature,
            max_tokens=self.profile.max_tokens,
        )
        latency_s = time.perf_counter() - started
        if not completion.choices:
            raise LLMError(f"{self.name}: the provider returned no reply")
        choice = completion.choices[0]
        usage = completion.usage
        return LLMResponse(
            profile=self.name,
            provider=self.provider_name,
            model=completion.model or self.profile.model,
            text=choice.message.content or "",
            finish_reason=choice.finish_reason or "unknown",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_s=round(latency_s, 3),
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
