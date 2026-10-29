"""Shared pytest fixtures, available to every test file without importing them.

`test_db_url` gives a test an empty throwaway database (named POSTGRES_DB + "_test") in the
project's Postgres container, and drops it afterwards, so tests never touch real data. Tests
that use it are skipped when Postgres isn't running.

LangSmith tracing is off for every test, even when .env turns it on, so tests never upload
anything. `traces` turns it on for one test and catches the traces in memory.

pytest loads this file on its own when you run the tests (`pytest`); it isn't run directly.
"""

import os
from collections.abc import Iterator

import langsmith
import psycopg
import pytest
from langsmith.run_helpers import tracing_context
from psycopg import sql
from psycopg.rows import TupleRow

from pr_reviewer.core.config import ConfigError, load_env
from pr_reviewer.db.connection import database_url
from tests.fakes import FakeLangSmith


@pytest.fixture(autouse=True, scope="session")
def _no_tracing() -> None:
    """Turn LangSmith tracing off for the whole test run, whatever .env says."""
    langsmith.configure(enabled=False)


@pytest.fixture
def traces() -> Iterator[FakeLangSmith]:
    """Turn tracing on for one test, catching the traces in memory instead of uploading them.

    Yields:
        FakeLangSmith: The traced steps, filled in as the test runs.
    """
    fake = FakeLangSmith()
    with tracing_context(enabled=True, client=fake):
        yield fake


def _drop_database(admin: psycopg.Connection[TupleRow], name: str) -> None:
    """Drop a database, closing any connections still open to it.

    Args:
        admin (psycopg.Connection[TupleRow]): Autocommit connection to a different database.
        name (str): Name of the database to drop.
    """
    admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


@pytest.fixture
def test_db_url() -> Iterator[str]:
    """Create an empty throwaway database and yield its connection string.

    The database lives in the project's Postgres container but has its own name
    (POSTGRES_DB + "_test"), so tests never touch real data. It is dropped after the test.
    The test is skipped when the settings are missing or Postgres isn't running.

    Yields:
        str: Connection string for the empty test database.
    """
    load_env()
    try:
        admin_url = database_url()
    except ConfigError as exc:
        pytest.skip(f"Postgres settings missing: {exc}")
    name = f"{os.environ['POSTGRES_DB']}_test"
    try:
        admin = psycopg.connect(admin_url, autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL is not running (start it with: docker compose up -d)")
    with admin:
        _drop_database(admin, name)
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    yield database_url(dbname=name)
    with psycopg.connect(admin_url, autocommit=True) as admin:
        _drop_database(admin, name)
