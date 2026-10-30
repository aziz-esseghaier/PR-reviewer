"""LLM response cache stored in PostgreSQL.

Saves every model reply in the llm_cache table (created by migration 0002), under a
fingerprint of the request: provider, model, messages, settings and sample number. LLMClient
looks here before calling a provider, so asking the same thing twice costs one API call.

The cache never expires. Re-running an experiment therefore gives exactly the same answers and
uses no quota, even though providers aren't fully repeatable at temperature 0.

Other modules import it; the file isn't run directly.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from pr_reviewer.core.config import ConfigError
from pr_reviewer.core.llm import LLMRequest, LLMResponse
from pr_reviewer.db.connection import table_exists

# Part of every fingerprint. Raising it makes all old entries unreachable, which is the way to
# start fresh if the fingerprint recipe ever changes.
KEY_VERSION = 1


def request_key(request: LLMRequest) -> str:
    """Compute the fingerprint the cache stores a request's reply under.

    The request is written as JSON with sorted keys and no spaces, so the same request always
    gives the same text, and then hashed with SHA-256. Fields left at None, such as no reply
    format, are left out, so adding such a field never changes the keys of older requests.

    Args:
        request (LLMRequest): The request to fingerprint.

    Returns:
        str: 64 hex characters. Any change to the request gives a different value.
    """
    canonical = json.dumps(
        request.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(f"v{KEY_VERSION}:{canonical}".encode()).hexdigest()


class PostgresCache:
    """Response cache kept in the llm_cache table. Pass it to LLMClient as `cache`."""

    def __init__(self, conn: psycopg.Connection[TupleRow]) -> None:
        """Use an open database connection for the cache.

        Args:
            conn (psycopg.Connection[TupleRow]): Open connection in autocommit mode.

        Raises:
            ConfigError: If the llm_cache table doesn't exist yet.
        """
        if not table_exists(conn, "llm_cache"):
            raise ConfigError("The llm_cache table is missing. Run: pr-reviewer db migrate")
        self._conn = conn

    def get(self, request: LLMRequest) -> LLMResponse | None:
        """Look up the saved reply to a request.

        Args:
            request (LLMRequest): The request to look up.

        Returns:
            LLMResponse | None: The saved reply, or None on a cache miss.
        """
        row = self._conn.execute(
            "SELECT response FROM llm_cache WHERE cache_key = %s", (request_key(request),)
        ).fetchone()
        return LLMResponse.model_validate(row[0]) if row else None

    def put(self, request: LLMRequest, response: LLMResponse) -> None:
        """Save the reply to a request. If one is already saved, keep the older one.

        Args:
            request (LLMRequest): The request that was sent.
            response (LLMResponse): The provider's reply.
        """
        self._conn.execute(
            """
            INSERT INTO llm_cache (cache_key, provider, model, request, response)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (cache_key) DO NOTHING
            """,
            (
                request_key(request),
                request.provider,
                request.model,
                Jsonb(request.model_dump(mode="json")),
                Jsonb(response.model_dump(mode="json")),
            ),
        )


@dataclass(frozen=True)
class CacheStats:
    """How many replies the cache holds for one model.

    Attributes:
        provider (str): Provider name, such as "groq".
        model (str): Model ID.
        replies (int): Number of cached replies.
        size_bytes (int): Space the requests and replies take, in bytes.
        newest (datetime): When the most recent reply was saved.
    """

    provider: str
    model: str
    replies: int
    size_bytes: int
    newest: datetime


def cache_stats(conn: psycopg.Connection[TupleRow]) -> list[CacheStats]:
    """Count the cached replies per provider and model.

    Args:
        conn (psycopg.Connection[TupleRow]): Open database connection.

    Returns:
        list[CacheStats]: One entry per provider and model, sorted by provider then model.
            Empty if the cache is empty.

    Raises:
        ConfigError: If the llm_cache table doesn't exist yet.
    """
    if not table_exists(conn, "llm_cache"):
        raise ConfigError("The llm_cache table is missing. Run: pr-reviewer db migrate")
    rows = conn.execute(
        """
        SELECT provider, model, count(*),
               sum(pg_column_size(request) + pg_column_size(response)), max(created_at)
        FROM llm_cache
        GROUP BY provider, model
        ORDER BY provider, model
        """
    ).fetchall()
    return [
        CacheStats(provider, model, int(replies), int(size), newest)
        for provider, model, replies, size, newest in rows
    ]
