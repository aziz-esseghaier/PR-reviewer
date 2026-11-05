"""Tests for the PostgreSQL response cache in core/cache.py.

The fingerprint tests run anywhere. The tests marked `integration` use a throwaway database
in the project's Postgres container and are skipped when it isn't running. The provider is
always fake, so no test uses API quota.

How to run (from the project folder):
    pytest tests/core/test_cache.py                      all cache tests
    pytest tests/core/test_cache.py -m "not integration" without the database
"""

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from psycopg.rows import TupleRow

from pr_reviewer.core.cache import PostgresCache, cache_stats, request_key
from pr_reviewer.core.config import ConfigError
from pr_reviewer.core.llm import LLMRequest, LLMResponse
from pr_reviewer.db.migrate import migrate
from tests.fakes import FakeProvider, fake_client

REQUEST = LLMRequest(
    provider="fake",
    model="fake-model-v1",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.0,
    max_tokens=64,
)
RESPONSE = LLMResponse(
    profile="fake-small",
    provider="fake",
    model="fake-model-v1",
    text="Hi there.",
    finish_reason="stop",
    input_tokens=12,
    output_tokens=3,
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


def test_request_key_is_a_stable_sha256() -> None:
    """Check that the same request always gives the same 64-character fingerprint."""
    copy = LLMRequest.model_validate(REQUEST.model_dump())
    assert request_key(REQUEST) == request_key(copy)
    assert len(request_key(REQUEST)) == 64


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "other"},
        {"model": "fake-model-v2"},
        {"messages": [{"role": "user", "content": "Hello!"}]},
        {"messages": [{"role": "system", "content": "Hello"}]},
        {"temperature": 0.7},
        {"max_tokens": 65},
        {"sample": 1},
        {"response_format": {"type": "json_object"}},
        {"reasoning_effort": "medium"},
    ],
)
def test_request_key_changes_with_any_field(change: dict[str, Any]) -> None:
    """Check that changing any part of the request gives a different fingerprint."""
    changed = REQUEST.model_copy(update=change)
    assert request_key(changed) != request_key(REQUEST)


def test_request_key_of_older_requests_is_unchanged() -> None:
    """Check that a request without a reply format keeps the key it had before formats existed.

    The expected value was computed with the recipe from before the response_format field,
    so replies already in the cache stay reachable.
    """
    assert request_key(REQUEST) == (
        "b18da43bcdee8a58a301cc0d47fb150165b63d6cf2a251648037ba9751697387"
    )


@pytest.mark.integration
def test_cache_needs_the_migration(test_db_url: str) -> None:
    """Check that using the cache before `db migrate` explains what to run."""
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        with pytest.raises(ConfigError, match="db migrate"):
            PostgresCache(conn)
        with pytest.raises(ConfigError, match="db migrate"):
            cache_stats(conn)


@pytest.mark.integration
def test_put_then_get_returns_the_same_reply(migrated_conn: psycopg.Connection[TupleRow]) -> None:
    """Check that a saved reply comes back unchanged, and an unknown request misses."""
    cache = PostgresCache(migrated_conn)
    assert cache.get(REQUEST) is None
    cache.put(REQUEST, RESPONSE)
    assert cache.get(REQUEST) == RESPONSE
    assert cache.get(REQUEST.model_copy(update={"sample": 1})) is None


@pytest.mark.integration
def test_saving_twice_keeps_the_first_reply(migrated_conn: psycopg.Connection[TupleRow]) -> None:
    """Check that a second save for the same request is ignored instead of failing."""
    cache = PostgresCache(migrated_conn)
    cache.put(REQUEST, RESPONSE)
    cache.put(REQUEST, RESPONSE.model_copy(update={"text": "Something else."}))
    saved = cache.get(REQUEST)
    assert saved is not None
    assert saved.text == "Hi there."


@pytest.mark.integration
def test_client_with_postgres_cache_asks_once(migrated_conn: psycopg.Connection[TupleRow]) -> None:
    """Check the whole path: the second identical call is answered from Postgres."""
    provider = FakeProvider("Hi there.")
    client = fake_client(provider, cache=PostgresCache(migrated_conn))

    first = client.complete([{"role": "user", "content": "Hello"}])
    second = client.complete([{"role": "user", "content": "Hello"}])

    assert len(provider.requests) == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.text == first.text


@pytest.mark.integration
def test_cache_stats_counts_per_model(migrated_conn: psycopg.Connection[TupleRow]) -> None:
    """Check that the stats count replies per provider and model."""
    assert cache_stats(migrated_conn) == []
    cache = PostgresCache(migrated_conn)
    cache.put(REQUEST, RESPONSE)
    cache.put(REQUEST.model_copy(update={"sample": 1}), RESPONSE)
    [stats] = cache_stats(migrated_conn)
    assert (stats.provider, stats.model, stats.replies) == ("fake", "fake-model-v1", 2)
    assert stats.size_bytes > 0
