"""The judge: the model SWR-Bench's scorer asks to match review points to known issues.

The judge is gemini-3.5-flash-lite, chosen once (results/judge-choice/) and fixed for the whole
project, so every result is scored the same way. make_judge() builds its client from the
judge profile in configs/models.yaml and refuses a profile whose model or settings changed.

JudgeChat stands in for SWR-Bench's own run_chat (see benchmark/official.py). It sends the
scorer's requests through our LLM client, so judge replies are cached, kept under the rate
limits and counted in the usage log like any other call.

The scorer retries a PR up to 3 times when the judge's JSON fails its checks, sending the exact
same request each time. The cache would hand back the same bad reply, so JudgeChat counts
repeats of a request and asks for a new sample each time (sample 1, 2, ...). A new JudgeChat
counts from 0 again, so a later run gets the same replies from the cache and the same scores.

Imported by other modules; the file isn't run directly.
"""

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import openai

from pr_reviewer.core.config import ConfigError, ModelsConfig
from pr_reviewer.core.llm import (
    ChatMessage,
    LLMClient,
    LLMError,
    LLMResponse,
    ResponseCache,
    explain_api_error,
)
from pr_reviewer.core.usage import UsageLog

log = logging.getLogger(__name__)

# The judge, fixed for the whole project. Temperature and max_tokens are the defaults of
# SWR-Bench's scorer (its --temperature and --max-tokens arguments).
JUDGE_PROFILE = "judge-gemini-3.5-flash-lite"
JUDGE_PROVIDER = "google"
JUDGE_MODEL = "gemini-3.5-flash-lite"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 8192


def make_judge(
    config: ModelsConfig,
    *,
    cache: ResponseCache | None = None,
    usage: UsageLog | None = None,
) -> LLMClient:
    """Create the client for the project's judge, after checking its profile hasn't changed.

    Args:
        config (ModelsConfig): The loaded model config.
        cache (ResponseCache | None): Where to look up and save judge replies.
        usage (UsageLog | None): Where to record requests and check the daily quota.

    Returns:
        LLMClient: The judge model.

    Raises:
        ConfigError: If the judge profile is missing, uses another provider or model, or has
            other settings than SWR-Bench's defaults, or its API key isn't set.
    """
    profile = config.profile(JUDGE_PROFILE)
    expected = (JUDGE_PROVIDER, JUDGE_MODEL, JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS)
    if (profile.provider, profile.model, profile.temperature, profile.max_tokens) != expected:
        raise ConfigError(
            f"The judge profile {JUDGE_PROFILE} must use {JUDGE_PROVIDER} {JUDGE_MODEL} at "
            f"temperature {JUDGE_TEMPERATURE} with max_tokens {JUDGE_MAX_TOKENS}. The judge "
            "stays the same for the whole project, so results stay comparable."
        )
    return LLMClient(JUDGE_PROFILE, config, cache=cache, usage=usage)


def scorer_settings(client: LLMClient) -> SimpleNamespace:
    """Give the settings SWR-Bench's scorer passes to run_chat, taken from the judge's profile.

    The scorer reads them from its command-line arguments (args.model, args.temperature,
    args.max_tokens). Taking them from the profile keeps one source for them.

    Args:
        client (LLMClient): The judge model.

    Returns:
        SimpleNamespace: An object with model, temperature and max_tokens, shaped like the
        scorer's parsed arguments.
    """
    profile = client.profile
    return SimpleNamespace(
        model=profile.model, temperature=profile.temperature, max_tokens=profile.max_tokens
    )


class JudgeChat:
    """SWR-Bench's run_chat, answered by one model profile through our LLM client.

    Use one JudgeChat per scoring run, from one thread.

    Attributes:
        client (LLMClient): The judge model.
        calls (int): How many times the scorer called it.
        responses (list[LLMResponse]): Every reply it returned, oldest first.
    """

    def __init__(self, client: LLMClient) -> None:
        """Wrap a judge model.

        Args:
            client (LLMClient): The judge model, with a cache and usage log in real runs.
        """
        self.client = client
        self.calls = 0
        self.responses: list[LLMResponse] = []
        self._seen: Counter[str] = Counter()

    def __call__(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float = 0.6,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        max_retries: int = 15,
    ) -> str | None:
        """Answer one scorer request, with the same arguments as SWR-Bench's run_chat.

        Args:
            model (str): Model ID the scorer asks for. Must be the judge's.
            messages (Sequence[ChatMessage]): The judge prompt.
            temperature (float): Must equal the judge profile's temperature.
            max_tokens (int | None): Must equal the judge profile's max_tokens.
            response_format (dict[str, Any] | None): The JSON format the reply must follow.
            max_retries (int): Ignored: our client has its own retry rules.

        Returns:
            str | None: The reply text, or None when the call failed, which makes the scorer
            try again.

        Raises:
            ValueError: If the scorer asks for another model or other settings than the
                judge profile's.
            LimitError: If a daily quota is used up, which stops the run.
        """
        settings = scorer_settings(self.client)
        if (model, temperature, max_tokens) != (
            settings.model,
            settings.temperature,
            settings.max_tokens,
        ):
            raise ValueError(
                f"The scorer asked for {model} (temperature {temperature}, max_tokens "
                f"{max_tokens}), not the judge profile {self.client.name}'s settings."
            )
        self.calls += 1
        key = json.dumps([list(messages), response_format], sort_keys=True)
        fingerprint = hashlib.sha256(key.encode()).hexdigest()
        sample = self._seen[fingerprint]
        self._seen[fingerprint] += 1
        try:
            response = self.client.complete(
                messages, sample=sample, response_format=response_format
            )
        except (openai.APIError, LLMError) as exc:
            reason = explain_api_error(exc) if isinstance(exc, openai.APIError) else str(exc)
            log.warning("%s: judge call failed: %s", self.client.name, reason.splitlines()[0])
            return None
        self.responses.append(response)
        return response.text
