"""Retries, throttling and daily limits, so LLM calls survive free-tier limits.

Three parts:
- Retries (`call_with_retries`): when a provider answers 429 (rate limit), 5xx (overloaded or
  broken) or doesn't answer in time, wait and try again. The wait is the provider's own hint
  when it gives one (Groq's retry-after header, Google's retryDelay), otherwise it doubles
  after each failure, with some randomness. Errors that asking again can't fix, such as a bad
  key or an unknown model, fail at once.
- Throttle (`Throttle`): keeps a model under its requests-per-minute and tokens-per-minute
  limits by waiting before a request that would go over, so most 429s never happen.
- Daily limits (`daily_status`): from the usage log, works out how much of a daily quota is
  in use and how long until a request fits, following the provider's reset rule: all at once
  at a fixed time (Google), gradually over 24 hours (Groq), or the last 24 hours if unknown.

All of them take a clock object, so tests can use a fake clock and never really sleep.
Imported by core/llm.py; the file isn't run directly.
"""

import json
import logging
import math
import random
import re
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

import openai

from pr_reviewer.core.config import FixedReset, ResetRule

log = logging.getLogger(__name__)

DAY = timedelta(days=1)

# HTTP codes worth retrying: rate limit, then server errors and overload.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# Rough token estimate: code and English average 3 to 4 characters per token, so 3 errs high.
CHARS_PER_TOKEN = 3
# Extra tokens per message for role markers, plus setup text some models add on their own
# (gpt-oss on Groq adds about 70 tokens to every request).
TOKENS_PER_MESSAGE = 4
HIDDEN_PROMPT_TOKENS = 100


class LimitError(Exception):
    """Raised when waiting won't help: a daily quota is used up, or a request is bigger than
    a limit could ever allow."""


class Clock(Protocol):
    """Tells the time and waits. Real runs use SystemClock; tests use a fake one."""

    def now(self) -> float:
        """Return the current time in seconds. Only differences between calls matter.

        Returns:
            float: Seconds on a clock that never goes backwards.
        """
        ...

    def wall_time(self) -> datetime:
        """Return the current date and time, for daily limits and the usage log.

        Returns:
            datetime: The current time in UTC.
        """
        ...

    def sleep(self, seconds: float) -> None:
        """Wait for a number of seconds.

        Args:
            seconds (float): How long to wait.
        """
        ...


class SystemClock:
    """The real clock: monotonic time, the system date and time, and time.sleep."""

    def now(self) -> float:
        """Return the current monotonic time.

        Returns:
            float: Seconds since an arbitrary start point.
        """
        return time.monotonic()

    def wall_time(self) -> datetime:
        """Return the current date and time.

        Returns:
            datetime: The current time in UTC.
        """
        return datetime.now(UTC)

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
        max_wait_s (float): Longest single wait we accept, whether the provider's hint or a
            daily limit asks for it. A longer one usually means the daily quota is used up,
            so we stop instead.
    """

    max_attempts: int = 6
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0
    max_wait_s: float = 120.0


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
        LimitError: If the provider asks for a wait longer than max_wait_s.
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
            if hint is not None and hint > policy.max_wait_s:
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


@dataclass(frozen=True)
class DailyStatus:
    """Where a model stands against one daily limit.

    Attributes:
        used (int): Amount counting against the limit right now, under the provider's rule.
        limit (int): The daily limit.
        wait_s (float): Seconds until the asked-for amount fits. 0 if it fits now.
    """

    used: int
    limit: int
    wait_s: float


def last_reset(rule: FixedReset, now: datetime) -> datetime:
    """Find the most recent moment a fixed-time quota reset.

    Args:
        rule (FixedReset): The reset time and time zone.
        now (datetime): The current time (timezone-aware).

    Returns:
        datetime: The last reset at or before now, in the rule's time zone.
    """
    local_now = now.astimezone(rule.zone)
    reset = datetime.combine(local_now.date(), rule.at, tzinfo=rule.zone)
    if reset > local_now:
        reset = datetime.combine(local_now.date() - timedelta(days=1), rule.at, tzinfo=rule.zone)
    return reset


def next_reset(rule: FixedReset, now: datetime) -> datetime:
    """Find the next moment a fixed-time quota resets.

    Built from the local date, so a day with a daylight-saving change still resets at the
    right local time.

    Args:
        rule (FixedReset): The reset time and time zone.
        now (datetime): The current time (timezone-aware).

    Returns:
        datetime: The first reset after now, in the rule's time zone.
    """
    previous = last_reset(rule, now)
    return datetime.combine(previous.date() + timedelta(days=1), rule.at, tzinfo=rule.zone)


def daily_status(
    usage: Sequence[tuple[datetime, int]],
    limit: int,
    rule: ResetRule,
    now: datetime,
    needed: int = 0,
) -> DailyStatus:
    """Work out how much of a daily limit is in use, and how long until more fits.

    Args:
        usage (Sequence[tuple[datetime, int]]): (time, amount) of each past request, oldest
            first, covering at least the last 24 hours. The amount is 1 for a request limit,
            or the request's tokens for a token limit.
        limit (int): The daily limit.
        rule (ResetRule): How the provider's quota comes back.
        now (datetime): The current time (timezone-aware).
        needed (int): Amount the next request needs. 0 just reports the current use.

    Returns:
        DailyStatus: The amount in use, the limit, and the seconds until `needed` fits.
    """
    if rule == "gradual":
        return _gradual_status(usage, limit, now, needed)
    if isinstance(rule, FixedReset):
        # Everything since the last reset counts; the whole quota comes back at the next one.
        since = last_reset(rule, now)
        used = sum(amount for when, amount in usage if when >= since)
        fits = used + needed <= limit
        wait_s = 0.0 if fits else (next_reset(rule, now) - now).total_seconds()
        return DailyStatus(used=used, limit=limit, wait_s=wait_s)
    # Rolling: everything in the last 24 hours counts; each request frees up 24 hours later.
    recent = [(when, amount) for when, amount in usage if when > now - DAY]
    used = sum(amount for _, amount in recent)
    excess = used + needed - limit
    wait_s = 0.0
    if excess > 0:
        freed = 0
        for when, amount in recent:
            freed += amount
            if freed >= excess:
                wait_s = (when + DAY - now).total_seconds()
                break
    return DailyStatus(used=used, limit=limit, wait_s=wait_s)


def _gradual_status(
    usage: Sequence[tuple[datetime, int]], limit: int, now: datetime, needed: int
) -> DailyStatus:
    """Daily status for a quota that refills a little at a time, like a bucket.

    The bucket holds `limit` and refills at limit / 24 h. Each request takes its amount out.
    Any use older than 24 hours has fully refilled, so we start 24 hours ago with a full
    bucket and replay the requests since then.

    Args:
        usage (Sequence[tuple[datetime, int]]): (time, amount) of past requests, oldest first.
        limit (int): The daily limit, which is also the bucket's size.
        now (datetime): The current time (timezone-aware).
        needed (int): Amount the next request needs.

    Returns:
        DailyStatus: The amount in use, the limit, and the seconds until `needed` fits.
    """
    rate_per_s = limit / DAY.total_seconds()
    start = now - DAY
    level = float(limit)
    previous = start
    for when, amount in usage:
        if when <= start:
            continue
        level = min(limit, level + (when - previous).total_seconds() * rate_per_s) - amount
        previous = when
    level = min(limit, level + (now - previous).total_seconds() * rate_per_s)
    wait_s = 0.0 if level >= needed else (needed - level) / rate_per_s
    return DailyStatus(used=round(limit - level), limit=limit, wait_s=wait_s)


def describe_wait(seconds: float, now: datetime) -> str:
    """Say when a wait ends, in words.

    Args:
        seconds (float): Length of the wait.
        now (datetime): The current time (timezone-aware).

    Returns:
        str: Such as "in about 2 h 05 min (Thu 09:00 your time)".
    """
    minutes = math.ceil(seconds / 60)
    hours, minutes = divmod(minutes, 60)
    span = f"{hours} h {minutes:02d} min" if hours else f"{minutes} min"
    ends = (now + timedelta(seconds=seconds)).astimezone()  # the computer's own time zone
    return f"in about {span} ({ends:%a %H:%M} your time)"
