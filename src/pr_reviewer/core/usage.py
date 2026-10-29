"""LLM usage log stored in PostgreSQL: one row per request sent, plus one per cache hit.

Writes to the llm_usage table (created by migration 0003) and answers two questions:
- which requests a model made recently, so the client can check its daily limits;
- how many requests, tokens, failures and cache hits there were per day and model, read
  from the llm_usage_daily view, for the `pr-reviewer usage` report.

Every try counts, failed ones included, because each one is a real request to the provider.
Cache hits are logged to show what the cache saved, but they never count against a limit.
Other modules import it; the file isn't run directly.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Protocol

import openai
import psycopg
from psycopg.rows import TupleRow

from pr_reviewer.core.config import ConfigError
from pr_reviewer.db.connection import table_exists

# What happened to a request. Matches the CHECK rule on llm_usage.outcome.
Outcome = Literal[
    "ok", "cache_hit", "rate_limited", "server_error", "timeout", "connection_error", "rejected"
]


@dataclass(frozen=True)
class UsageEntry:
    """One request sent to a provider, or one reply served from the cache.

    Attributes:
        time (datetime): When the request was sent (UTC).
        provider (str): Provider name, such as "groq".
        model (str): Model ID.
        profile (str): Profile name that made the request.
        outcome (Outcome): "ok", "cache_hit" or the kind of failure.
        status_code (int | None): HTTP status. None for cache hits, timeouts and dropped
            connections.
        input_tokens (int): Prompt tokens. 0 when the request failed.
        output_tokens (int): Reply tokens. 0 when the request failed.
        latency_s (float | None): Seconds the request took. None for cache hits.
    """

    time: datetime
    provider: str
    model: str
    profile: str
    outcome: Outcome
    status_code: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float | None = None


class UsageLog(Protocol):
    """Anything that can record requests and list a model's recent ones.

    LLMClient only needs these two methods, so real runs pass the PostgreSQL log and tests
    pass a simple in-memory one.
    """

    def record(self, entry: UsageEntry) -> None:
        """Save one entry.

        Args:
            entry (UsageEntry): The request or cache hit to save.
        """
        ...

    def recent(self, provider: str, model: str, since: datetime) -> list[tuple[datetime, int]]:
        """List a model's requests since a moment, without cache hits.

        Args:
            provider (str): Provider name.
            model (str): Model ID.
            since (datetime): Only requests after this moment count.

        Returns:
            list[tuple[datetime, int]]: (time, tokens) of each request, oldest first.
        """
        ...


def outcome_of(exc: openai.APIError) -> tuple[Outcome, int | None]:
    """Classify a failed request for the usage log.

    Args:
        exc (openai.APIError): The error the SDK raised.

    Returns:
        tuple[Outcome, int | None]: The outcome and the HTTP status, if there was one.
    """
    if isinstance(exc, openai.APITimeoutError):
        return "timeout", None
    if isinstance(exc, openai.APIConnectionError):
        return "connection_error", None
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code == 429:
            return "rate_limited", 429
        if exc.status_code >= 500:
            return "server_error", exc.status_code
        return "rejected", exc.status_code
    return "rejected", None


class PostgresUsageLog:
    """Usage log kept in the llm_usage table. Pass it to LLMClient as `usage`."""

    def __init__(self, conn: psycopg.Connection[TupleRow]) -> None:
        """Use an open database connection for the log.

        Args:
            conn (psycopg.Connection[TupleRow]): Open connection in autocommit mode.

        Raises:
            ConfigError: If the llm_usage table doesn't exist yet.
        """
        _require_table(conn)
        self._conn = conn

    def record(self, entry: UsageEntry) -> None:
        """Save one entry.

        Args:
            entry (UsageEntry): The request or cache hit to save.
        """
        self._conn.execute(
            """
            INSERT INTO llm_usage (created_at, provider, model, profile, outcome, status_code,
                                   input_tokens, output_tokens, latency_s)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                entry.time,
                entry.provider,
                entry.model,
                entry.profile,
                entry.outcome,
                entry.status_code,
                entry.input_tokens,
                entry.output_tokens,
                entry.latency_s,
            ),
        )

    def recent(self, provider: str, model: str, since: datetime) -> list[tuple[datetime, int]]:
        """List a model's requests since a moment, without cache hits.

        Args:
            provider (str): Provider name.
            model (str): Model ID.
            since (datetime): Only requests after this moment count.

        Returns:
            list[tuple[datetime, int]]: (time, tokens) of each request, oldest first.
        """
        rows = self._conn.execute(
            """
            SELECT created_at, input_tokens + output_tokens
            FROM llm_usage
            WHERE provider = %s AND model = %s AND outcome <> 'cache_hit' AND created_at > %s
            ORDER BY created_at
            """,
            (provider, model, since),
        ).fetchall()
        return [(when, int(tokens)) for when, tokens in rows]


@dataclass(frozen=True)
class DailyUsage:
    """One day of use for one model, from the llm_usage_daily view.

    Attributes:
        day (date): The day, in UTC.
        provider (str): Provider name.
        model (str): Model ID.
        requests (int): Requests sent, failed ones included.
        failed (int): Requests that failed.
        tokens (int): Tokens of the requests sent.
        cache_hits (int): Replies served from the cache instead.
        tokens_saved (int): Tokens those cache hits would have cost.
    """

    day: date
    provider: str
    model: str
    requests: int
    failed: int
    tokens: int
    cache_hits: int
    tokens_saved: int


def usage_by_day(conn: psycopg.Connection[TupleRow], days: int = 7) -> list[DailyUsage]:
    """Read the daily totals per model for the last few days.

    Args:
        conn (psycopg.Connection[TupleRow]): Open database connection.
        days (int): How many days back to go, today included.

    Returns:
        list[DailyUsage]: One entry per day and model, newest day first.

    Raises:
        ConfigError: If the llm_usage table doesn't exist yet.
    """
    _require_table(conn)
    rows = conn.execute(
        """
        SELECT day, provider, model, requests, failed, tokens, cache_hits, tokens_saved
        FROM llm_usage_daily
        WHERE day > (now() AT TIME ZONE 'UTC')::date - %s
        ORDER BY day DESC, provider, model
        """,
        (days,),
    ).fetchall()
    return [DailyUsage(*(row[:3]), *(int(value) for value in row[3:])) for row in rows]


def _require_table(conn: psycopg.Connection[TupleRow]) -> None:
    """Make sure the usage table exists.

    Args:
        conn (psycopg.Connection[TupleRow]): Open database connection.

    Raises:
        ConfigError: If the llm_usage table doesn't exist yet.
    """
    if not table_exists(conn, "llm_usage"):
        raise ConfigError("The llm_usage table is missing. Run: pr-reviewer db migrate")
