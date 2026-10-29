"""Tests for retries, throttling and daily limits in core/limits.py.

Checks which errors are retried, how long the waits are (the provider's hint or doubling
backoff), when the retries give up, that the throttle keeps requests and tokens under their
per-minute limits, and that daily quotas are counted the way each provider resets them: at a
fixed time (including across a daylight-saving change), gradually, or over the last 24 hours.
A fake clock stands in for real time, so nothing actually waits.

How to run (from the project folder):
    pytest tests/core/test_limits.py
"""

import random
from collections.abc import Callable
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import httpx2
import openai
import pytest

from pr_reviewer.core.config import FixedReset
from pr_reviewer.core.limits import (
    LimitError,
    RetryPolicy,
    Throttle,
    backoff_delay,
    call_with_retries,
    daily_status,
    estimate_tokens,
    hinted_delay,
    is_retryable,
    last_reset,
    next_reset,
)
from tests.fakes import FakeClock

MIDNIGHT_PACIFIC = FixedReset(at=time(0, 0), zone=ZoneInfo("America/Los_Angeles"))
HOUR = timedelta(hours=1)

REQUEST = httpx2.Request("POST", "https://llm.example.test/v1/chat/completions")


def status_error(
    status: int, headers: dict[str, str] | None = None, body: object = None
) -> openai.APIStatusError:
    """Build the error the SDK raises when a provider answers with an HTTP error code.

    Args:
        status (int): HTTP status code, such as 503.
        headers (dict[str, str] | None): Response headers, such as retry-after.
        body (object): Parsed error body. None uses a minimal OpenAI-style one.

    Returns:
        openai.APIStatusError: The error.
    """
    body = body if body is not None else {"error": {"message": "x"}}
    response = httpx2.Response(status, headers=headers, json=body, request=REQUEST)
    return openai.APIStatusError(f"Error code: {status}", response=response, body=body)


def failing_then(
    errors: list[openai.APIError], value: str = "ok"
) -> tuple[list[int], Callable[[], str]]:
    """Make a send function that raises the given errors in order, then returns a value.

    Args:
        errors (list[openai.APIError]): Errors for the first tries.
        value (str): What the first try after the errors returns.

    Returns:
        tuple[list[int], Callable[[], str]]: A one-item list counting the tries, and the
            send function.
    """
    tries = [0]

    def send() -> str:
        tries[0] += 1
        if errors:
            raise errors.pop(0)
        return value

    return tries, send


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_temporary_http_errors_are_retryable(status: int) -> None:
    """Check that rate limits and server errors are worth another try."""
    assert is_retryable(status_error(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_http_errors_are_not_retryable(status: int) -> None:
    """Check that a bad request, key or model fails at once."""
    assert not is_retryable(status_error(status))


def test_timeouts_and_dropped_connections_are_retryable() -> None:
    """Check that no answer, or a broken connection, is worth another try."""
    assert is_retryable(openai.APITimeoutError(request=REQUEST))
    assert is_retryable(openai.APIConnectionError(request=REQUEST))


def test_hint_from_retry_after_header() -> None:
    """Check that Groq-style retry-after headers are read in seconds."""
    assert hinted_delay(status_error(429, headers={"retry-after": "7"})) == 7.0


def test_hint_from_google_retry_delay() -> None:
    """Check that Google's retryDelay inside its list-wrapped error body is read."""
    body = [
        {
            "error": {
                "code": 429,
                "details": [{"@type": "type.googleapis.com/RetryInfo", "retryDelay": "27s"}],
            }
        }
    ]
    assert hinted_delay(status_error(429, body=body)) == 27.0


def test_no_hint_gives_none() -> None:
    """Check that an error without a hint, or with a date-style header, gives None."""
    assert hinted_delay(status_error(503)) is None
    date_header = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
    assert hinted_delay(status_error(503, headers=date_header)) is None
    assert hinted_delay(openai.APITimeoutError(request=REQUEST)) is None


def test_backoff_doubles_with_randomness_and_a_cap() -> None:
    """Check that waits lie between half and all of 2, 4, 8, ... seconds, capped at 60."""
    policy = RetryPolicy()
    rng = random.Random(0)
    for attempt, ceiling in [(1, 2), (2, 4), (3, 8), (6, 60), (9, 60)]:
        delay = backoff_delay(attempt, policy, rng)
        assert ceiling / 2 <= delay <= ceiling


def test_retries_until_success() -> None:
    """Check that two overloads are waited out and the third try's result comes back."""
    clock = FakeClock()
    tries, send = failing_then([status_error(503), status_error(503)])
    result = call_with_retries(send, label="test", policy=RetryPolicy(), clock=clock)
    assert result.value == "ok"
    assert result.attempts == 3 == tries[0]
    assert len(clock.sleeps) == 2
    assert result.waited_s == pytest.approx(sum(clock.sleeps))


def test_provider_hint_sets_the_wait() -> None:
    """Check that the provider's own retry-after is used instead of our backoff."""
    clock = FakeClock()
    _, send = failing_then([status_error(429, headers={"retry-after": "12"})])
    call_with_retries(send, label="test", policy=RetryPolicy(), clock=clock)
    assert clock.sleeps == [12.0]


def test_permanent_error_is_not_retried() -> None:
    """Check that a rejected key fails on the first try, without waiting."""
    clock = FakeClock()
    tries, send = failing_then([status_error(401)])
    with pytest.raises(openai.APIStatusError):
        call_with_retries(send, label="test", policy=RetryPolicy(), clock=clock)
    assert tries[0] == 1
    assert clock.sleeps == []


def test_gives_up_after_max_attempts() -> None:
    """Check that the last error is raised once every try has failed."""
    clock = FakeClock()
    policy = RetryPolicy(max_attempts=4)
    tries, send = failing_then([status_error(503) for _ in range(10)])
    with pytest.raises(openai.APIStatusError):
        call_with_retries(send, label="test", policy=policy, clock=clock)
    assert tries[0] == 4
    assert len(clock.sleeps) == 3


def test_very_long_hint_stops_instead_of_sleeping() -> None:
    """Check that a request to wait for hours (a used-up daily quota) stops at once."""
    clock = FakeClock()
    _, send = failing_then([status_error(429, headers={"retry-after": "3600"})])
    with pytest.raises(LimitError, match="daily quota"):
        call_with_retries(send, label="test", policy=RetryPolicy(), clock=clock)
    assert clock.sleeps == []


def test_estimate_includes_the_full_reply_budget() -> None:
    """Check that the estimate grows with the prompt and always counts max_tokens."""
    short = estimate_tokens(["Hello"], max_tokens=500)
    long = estimate_tokens(["Hello" * 300], max_tokens=500)
    assert short > 500
    assert long > short


def test_throttle_without_limits_never_waits() -> None:
    """Check that a model with no limits is never slowed down."""
    clock = FakeClock()
    throttle = Throttle(None, None, clock=clock, label="test")
    for _ in range(100):
        throttle.reserve(10_000)
    assert clock.sleeps == []


def test_throttle_keeps_requests_per_minute() -> None:
    """Check that the third request in a 2-per-minute window waits for the oldest to expire."""
    clock = FakeClock()
    throttle = Throttle(2, None, clock=clock, label="test")
    throttle.reserve(1)
    clock.time = 30.0
    throttle.reserve(1)
    _, waited = throttle.reserve(1)
    assert waited == pytest.approx(30.0)  # the first request leaves the window at t=60


def test_throttle_keeps_tokens_per_minute() -> None:
    """Check that a request that would pass the token limit waits for room."""
    clock = FakeClock()
    throttle = Throttle(None, 1000, clock=clock, label="test")
    throttle.reserve(600)
    _, waited = throttle.reserve(600)
    assert waited == pytest.approx(60.0)


def test_throttle_uses_the_real_token_count() -> None:
    """Check that replacing the estimate with the real count frees room at once."""
    clock = FakeClock()
    throttle = Throttle(None, 1000, clock=clock, label="test")
    slot, _ = throttle.reserve(900)
    slot.tokens = 100  # the reply was much shorter than max_tokens
    _, waited = throttle.reserve(800)
    assert waited == 0.0


def test_throttle_refuses_a_request_bigger_than_the_budget() -> None:
    """Check that a request that could never fit stops with a clear message."""
    throttle = Throttle(None, 1000, clock=FakeClock(), label="test")
    with pytest.raises(LimitError, match="Shorten the prompt"):
        throttle.reserve(1500)


def test_rolling_window_frees_the_oldest_request_after_24_hours() -> None:
    """Check that with no known reset rule, room comes back 24 h after the oldest request."""
    now = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
    usage = [(now - 23 * HOUR, 1), (now - HOUR, 1)]
    status = daily_status(usage, limit=2, rule="rolling", now=now, needed=1)
    assert status.used == 2
    assert status.wait_s == pytest.approx(HOUR.total_seconds())


def test_fixed_reset_counts_only_since_the_last_reset() -> None:
    """Check that requests before midnight Pacific don't count, and room returns at the next."""
    now = datetime(2026, 10, 31, 20, 0, tzinfo=UTC)  # 13:00 in Los Angeles (PDT, UTC-7)
    before_reset = datetime(2026, 10, 31, 6, 0, tzinfo=UTC)  # 23:00 the day before, in LA
    after_reset = datetime(2026, 10, 31, 8, 0, tzinfo=UTC)
    status = daily_status(
        [(before_reset, 1), (after_reset, 1)], limit=1, rule=MIDNIGHT_PACIFIC, now=now, needed=1
    )
    assert status.used == 1
    # The next reset is midnight on 1 November in LA, which is 07:00 UTC.
    assert status.wait_s == pytest.approx(11 * HOUR.total_seconds())


def test_fixed_reset_follows_daylight_saving() -> None:
    """Check the reset times around the end of daylight saving in LA on 1 November 2026."""
    now = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)  # 12:00 in LA, after clocks went back
    assert last_reset(MIDNIGHT_PACIFIC, now) == datetime(2026, 11, 1, 7, 0, tzinfo=UTC)  # PDT
    assert next_reset(MIDNIGHT_PACIFIC, now) == datetime(2026, 11, 2, 8, 0, tzinfo=UTC)  # PST


def test_gradual_quota_refills_bit_by_bit() -> None:
    """Check that a quota of 24 a day gives one request back every hour after a burst."""
    now = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
    burst = [(now - timedelta(seconds=1), 1)] * 24
    status = daily_status(burst, limit=24, rule="gradual", now=now, needed=1)
    assert status.used == 24
    assert status.wait_s == pytest.approx(HOUR.total_seconds(), abs=2)

    two_hours_later = now + 2 * HOUR
    status = daily_status(burst, limit=24, rule="gradual", now=two_hours_later, needed=1)
    assert status.used == 22
    assert status.wait_s == 0.0


def test_gradual_quota_forgets_use_older_than_a_day() -> None:
    """Check that requests from more than 24 hours ago have fully refilled."""
    now = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
    status = daily_status([(now - 25 * HOUR, 24)], limit=24, rule="gradual", now=now, needed=24)
    assert status.used == 0
    assert status.wait_s == 0.0
