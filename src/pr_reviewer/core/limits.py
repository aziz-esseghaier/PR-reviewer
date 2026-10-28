"""Retries and throttling, so LLM calls survive free-tier limits instead of crashing.

Two parts:
- Retries (`call_with_retries`): when a provider answers 429 (rate limit), 5xx (overloaded or
  broken) or doesn't answer in time, wait and try again. The wait is the provider's own hint
  when it gives one (Groq's retry-after header, Google's retryDelay), otherwise it doubles
  after each failure, with some randomness. Errors that asking again can't fix, such as a bad
  key or an unknown model, fail at once.
- Throttle (`Throttle`): keeps a model under its requests-per-minute and tokens-per-minute
  limits by waiting before a request that would go over, so most 429s never happen.

Both take a clock object, so tests can use a fake clock and never really sleep.
Imported by core/llm.py; the file isn't run directly.
"""

import json
import logging
import random
import re
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import openai

log = logging.getLogger(__name__)

# HTTP codes worth retrying: rate limit, then server errors and overload.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# Rough token estimate: code and English average 3 to 4 characters per token, so 3 errs high.
CHARS_PER_TOKEN = 3
# Extra tokens per message for role markers, plus setup text some models add on their own
# (gpt-oss on Groq adds about 70 tokens to every request).
TOKENS_PER_MESSAGE = 4
HIDDEN_PROMPT_TOKENS = 100


class LimitError(Exception):
    """Raised when waiting won't help: the provider wants a very long wait (usually the daily
    quota is used up), or a request is bigger than the model's whole per-minute token budget."""


class Clock(Protocol):
    """Tells the time and waits. Real runs use SystemClock; tests use a fake one."""

    def now(self) -> float:
        """Return the current time in seconds. Only differences between calls matter.

        Returns:
            float: Seconds on a clock that never goes backwards.
        """
        ...

    def sleep(self, seconds: float) -> None:
        """Wait for a number of seconds.

        Args:
            seconds (float): How long to wait.
        """
        ...


class SystemClock:
    """The real clock: monotonic time and time.sleep."""

    def now(self) -> float:
        """Return the current monotonic time.

        Returns:
            float: Seconds since an arbitrary start point.
        """
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        """Wait for a number of seconds.

        Args:
            seconds (float): How long to wait.
        """
        time.sleep(seconds)


@dataclass(frozen=True)
class RetryPolicy:
    """How patient to be with a failing provider.

    Attributes:
        max_attempts (int): Tries in total, including the first one.
        base_delay_s (float): Wait after the first failure, before randomness.
            It doubles after each further failure.
        max_delay_s (float): Longest wait between two tries, when the provider gives no hint.
        max_hinted_wait_s (float): Longest wait we accept from the provider's own hint.
            A longer hint usually means the daily quota is used up, so we stop instead.
    """

    max_attempts: int = 6
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0
    max_hinted_wait_s: float = 120.0


@dataclass(frozen=True)
class Retried[T]:
    """The result of a call that may have needed several tries.

    Attributes:
        value (T): What the successful try returned.
        attempts (int): Tries it took, 1 if the first one worked.
        waited_s (float): Seconds spent waiting between tries.
    """

    value: T
    attempts: int
    waited_s: float


def is_retryable(exc: openai.APIError) -> bool:
    """Decide whether asking again could fix an error.

    Args:
        exc (openai.APIError): The error the SDK raised.

    Returns:
        bool: True for rate limits, server errors, timeouts and dropped connections.
    """
    if isinstance(exc, openai.APIConnectionError):  # also covers timeouts
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code in RETRYABLE_STATUS


def hinted_delay(exc: openai.APIError) -> float | None:
    """Read how long the provider asked us to wait, if it said.

    Groq sends a retry-after header in seconds. Google puts a retryDelay such as "27s" in the
    error body.

    Args:
        exc (openai.APIError): The error the SDK raised.

    Returns:
        float | None: Seconds to wait, or None if the provider gave no hint.
    """
    if not isinstance(exc, openai.APIStatusError):
        return None
    header = exc.response.headers.get("retry-after")
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass  # the header can also be a date; we fall back to our own backoff then
    match = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', json.dumps(exc.body))
    return float(match.group(1)) if match else None


def backoff_delay(attempt: int, policy: RetryPolicy, rng: random.Random) -> float:
    """Pick the wait after a failed try when the provider gave no hint.

    The ceiling doubles with each failure (2, 4, 8, ... seconds, up to max_delay_s), and the
    wait is a random point between half the ceiling and the full ceiling. The randomness keeps
    several workers from retrying at the same moment.

    Args:
        attempt (int): Number of the try that just failed, starting at 1.
        policy (RetryPolicy): The retry settings.
        rng (random.Random): Source of randomness.

    Returns:
        float: Seconds to wait.
    """
    ceiling = min(policy.max_delay_s, policy.base_delay_s * 2.0 ** (attempt - 1))
    return ceiling * (0.5 + rng.random() / 2)


def _reason(exc: openai.APIError) -> str:
    """Describe an error in a few words for the log.

    Args:
        exc (openai.APIError): The error the SDK raised.

    Returns:
        str: Such as "HTTP 503 overloaded" or "no answer in time".
    """
    if isinstance(exc, openai.APITimeoutError):
        return "no answer in time"
    if isinstance(exc, openai.APIConnectionError):
        return "connection failed"
    if isinstance(exc, openai.APIStatusError):
        names = {429: "rate limit", 503: "overloaded", 504: "gateway timeout"}
        return f"HTTP {exc.status_code} {names.get(exc.status_code, 'server error')}"
    return type(exc).__name__


def call_with_retries[T](
    send: Callable[[], T],
    *,
    label: str,
    policy: RetryPolicy,
    clock: Clock,
    rng: random.Random | None = None,
) -> Retried[T]:
    """Call a function, and try again after temporary provider errors.

    Args:
        send (Callable[[], T]): Makes one try, such as one API request.
        label (str): Name for log lines, such as the model profile.
        policy (RetryPolicy): How many tries and how long to wait.
        clock (Clock): Used to wait between tries.
        rng (random.Random | None): Source of randomness for the waits. None uses a new one.

    Returns:
        Retried[T]: The successful result, the number of tries and the time spent waiting.

    Raises:
        openai.APIError: The error itself if it can't be fixed by asking again, or the last
            error once all tries are used up.
        LimitError: If the provider asks for a wait longer than max_hinted_wait_s.
    """
    rng = rng or random.Random()
    waited_s = 0.0
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return Retried(value=send(), attempts=attempt, waited_s=waited_s)
        except openai.APIError as exc:
            if not is_retryable(exc):
                raise
            if attempt == policy.max_attempts:
                log.warning("%s: %s, giving up after %d tries", label, _reason(exc), attempt)
                raise
            hint = hinted_delay(exc)
            if hint is not None and hint > policy.max_hinted_wait_s:
                raise LimitError(
                    f"{label}: the provider asks us to wait {hint:.0f} s, which usually means "
                    "the daily quota is used up. Try again later."
                ) from exc
            delay = hint if hint is not None else backoff_delay(attempt, policy, rng)
            log.warning(
                "%s: %s, waiting %.1f s (try %d of %d)",
                label,
                _reason(exc),
                delay,
                attempt,
                policy.max_attempts,
            )
            clock.sleep(delay)
            waited_s += delay
    raise AssertionError("unreachable: the loop returns or raises")


def estimate_tokens(texts: Sequence[str], max_tokens: int) -> int:
    """Estimate, on the high side, how many tokens a request will use against the limits.

    Providers such as Groq count the full max_tokens against the per-minute budget before the
    reply is written, so it is included in full.

    Args:
        texts (Sequence[str]): The text of each message.
        max_tokens (int): Longest reply allowed.

    Returns:
        int: Estimated prompt tokens plus max_tokens.
    """
    prompt = sum(len(text) for text in texts) // CHARS_PER_TOKEN
    return prompt + TOKENS_PER_MESSAGE * len(texts) + HIDDEN_PROMPT_TOKENS + max_tokens


@dataclass
class Slot:
    """One request the throttle has let through.

    Attributes:
        time (float): When it was let through, on the throttle's clock.
        tokens (int): Tokens it counts for. Starts as the estimate; set it to the real count
            once the reply is in.
    """

    time: float
    tokens: int


class Throttle:
    """Keeps one model under its per-minute request and token limits.

    It remembers the requests of the last 60 seconds. Before a new one, it waits until the new
    request fits under both limits. It only knows about requests from this process.
    """

    WINDOW_S = 60.0

    def __init__(
        self,
        requests_per_minute: int | None,
        tokens_per_minute: int | None,
        *,
        clock: Clock,
        label: str,
    ) -> None:
        """Set the limits to enforce.

        Args:
            requests_per_minute (int | None): Most requests in any 60 seconds. None means
                no limit.
            tokens_per_minute (int | None): Most tokens in any 60 seconds. None means no limit.
            clock (Clock): Used to read the time and to wait.
            label (str): Name for log lines, such as the model profile.
        """
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._clock = clock
        self._label = label
        self._slots: deque[Slot] = deque()

    def reserve(self, tokens: int) -> tuple[Slot, float]:
        """Wait until a request fits under the limits, then count it.

        Args:
            tokens (int): Estimated tokens of the request.

        Returns:
            tuple[Slot, float]: The slot for this request (update its tokens once the real
                count is known) and the seconds spent waiting.

        Raises:
            LimitError: If the request alone is bigger than the per-minute token limit.
        """
        if self._tpm is not None and tokens > self._tpm:
            raise LimitError(
                f"{self._label}: this request needs about {tokens} tokens, more than the "
                f"model's {self._tpm} tokens per minute. Shorten the prompt or lower max_tokens."
            )
        waited_s = 0.0
        while (delay := self._delay_for(tokens)) > 0:
            log.info("%s: waiting %.1f s to stay under the per-minute limits", self._label, delay)
            self._clock.sleep(delay)
            waited_s += delay
        slot = Slot(time=self._clock.now(), tokens=tokens)
        self._slots.append(slot)
        return slot, waited_s

    def _delay_for(self, tokens: int) -> float:
        """Work out how long to wait before a request of this size fits.

        Args:
            tokens (int): Estimated tokens of the request.

        Returns:
            float: Seconds to wait. 0 means it fits now.
        """
        now = self._clock.now()
        while self._slots and self._slots[0].time <= now - self.WINDOW_S:
            self._slots.popleft()
        delays = [0.0]
        if self._rpm is not None and len(self._slots) >= self._rpm:
            # Enough of the oldest requests must leave the window to make room for one more.
            must_leave = self._slots[len(self._slots) - self._rpm]
            delays.append(must_leave.time + self.WINDOW_S - now)
        if self._tpm is not None:
            excess = sum(slot.tokens for slot in self._slots) + tokens - self._tpm
            if excess > 0:
                # Wait until enough of the oldest requests leave the window to free the excess.
                freed = 0
                for slot in self._slots:
                    freed += slot.tokens
                    if freed >= excess:
                        delays.append(slot.time + self.WINDOW_S - now)
                        break
        return max(delays)
