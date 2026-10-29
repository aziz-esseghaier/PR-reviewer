"""Tests for the PostgreSQL usage log in core/usage.py.

Checks how failed requests are classified, that entries are saved and read back, that cache
hits and old requests are left out of the recent list, and that the daily view adds things up.
The tests marked `integration` use a throwaway database and are skipped when Postgres isn't
running.

How to run (from the project folder):
    pytest tests/core/test_usage.py                       all usage tests
    pytest tests/core/test_usage.py -m "not integration"  without the database
"""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import httpx2
import openai
import psycopg
import pytest
from psycopg.rows import TupleRow

from pr_reviewer.core.config import ConfigError
from pr_reviewer.core.usage import PostgresUsageLog, UsageEntry, outcome_of, usage_by_day
from pr_reviewer.db.migrate import migrate

REQUEST = httpx2.Request("POST", "https://llm.example.test/v1/chat/completions")
NOW = datetime.now(UTC)


def entry(outcome: str = "ok", minutes_ago: float = 1, tokens: int = 100) -> UsageEntry:
    """Build a usage entry for the fake model.

    Args:
        outcome (str): What happened to the request.
        minutes_ago (float): How long before now the request was sent.
        tokens (int): Its input tokens (output tokens are always 10).

    Returns:
        UsageEntry: The entry.
    """
    return UsageEntry(
        time=NOW - timedelta(minutes=minutes_ago),
        provider="fake",
        model="fake-model-v1",
        profile="fake-small",
        outcome=outcome,  # type: ignore[arg-type]
        status_code=200 if outcome == "ok" else None,
        input_tokens=tokens,
        output_tokens=10,
        latency_s=0.5,
    )


@pytest.fixture
def migrated_conn(test_db_url: str) -> Iterator[psycopg.Connection[TupleRow]]:
    """Open a connection to a fresh test database with all migrations applied.

    Args:
        test_db_url (str): Connection string of the empty throwaway database.

    Yields:
        psycopg.Connection[TupleRow]: Autocommit connection to the migrated database.
    """
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        migrate(conn)
        yield conn


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (openai.APITimeoutError(request=REQUEST), ("timeout", None)),
        (openai.APIConnectionError(request=REQUEST), ("connection_error", None)),
    ],
)
def test_outcome_of_network_problems(
    error: openai.APIError, expected: tuple[str, int | None]
) -> None:
    """Check that timeouts and dropped connections are told apart."""
    assert outcome_of(error) == expected


@pytest.mark.parametrize(
    ("status", "outcome"),
    [(429, "rate_limited"), (503, "server_error"), (500, "server_error"), (401, "rejected")],
)
def test_outcome_of_http_errors(status: int, outcome: str) -> None:
    """Check that HTTP errors are sorted into rate limits, server errors and rejections."""
    response = httpx2.Response(status, json={"error": {"message": "x"}}, request=REQUEST)
    error = openai.APIStatusError("x", response=response, body=None)
    assert outcome_of(error) == (outcome, status)


@pytest.mark.integration
def test_log_needs_the_migration(test_db_url: str) -> None:
    """Check that using the log before `db migrate` explains what to run."""
    with (
        psycopg.connect(test_db_url, autocommit=True) as conn,
        pytest.raises(ConfigError, match="db migrate"),
    ):
        PostgresUsageLog(conn)


@pytest.mark.integration
def test_recent_leaves_out_cache_hits_and_old_requests(
    migrated_conn: psycopg.Connection[TupleRow],
) -> None:
    """Check that only real requests after the cut-off come back, oldest first."""
    log = PostgresUsageLog(migrated_conn)
    log.record(entry(minutes_ago=30 * 60))  # 30 hours ago
    log.record(entry(minutes_ago=5, tokens=200))
    log.record(entry("rate_limited", minutes_ago=4, tokens=0))
    log.record(entry("cache_hit", minutes_ago=3))
    recent = log.recent("fake", "fake-model-v1", NOW - timedelta(days=1))
    assert [tokens for _, tokens in recent] == [210, 10]
    assert recent[0][0] < recent[1][0]


@pytest.mark.integration
def test_daily_view_adds_up(migrated_conn: psycopg.Connection[TupleRow]) -> None:
    """Check the daily totals: requests include failures, cache hits are counted apart."""
    log = PostgresUsageLog(migrated_conn)
    log.record(entry(tokens=100))
    log.record(entry("server_error", tokens=0))
    log.record(entry("cache_hit", tokens=100))
    [today] = usage_by_day(migrated_conn, days=1)
    assert (today.requests, today.failed, today.cache_hits) == (2, 1, 1)
    assert today.tokens == 110 + 10  # the failed request has 0 input and 10 output tokens
    assert today.tokens_saved == 110
