"""Tests for keeping retrieved context (context/store.py): files on disk, rows in PostgreSQL.

Checks that candidates are saved as a file and recorded under their settings key, read back,
replaced when retrieved again, kept apart from other settings, and treated as missing when
their file is gone. Marked `integration`: they use a throwaway database and are skipped when
Postgres isn't running.

How to run (from the project folder):
    pytest tests/context/test_store.py
"""

from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import TupleRow

from pr_reviewer.context.store import load_context, save_context, stored_contexts
from pr_reviewer.db.migrate import migrate
from tests.context.helpers import candidates, piece

pytestmark = pytest.mark.integration

PR_ID = "psf__requests-1"


@pytest.fixture
def migrated(test_db_url: str) -> Iterator[psycopg.Connection[TupleRow]]:
    """Open a fresh migrated test database with one benchmark PR.

    Args:
        test_db_url (str): Connection string of the empty throwaway database.

    Yields:
        psycopg.Connection[TupleRow]: Autocommit connection.
    """
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        migrate(conn)
        conn.execute(
            "INSERT INTO bench_prs VALUES (%s, 'psf/requests', 2022, 'clean', 'dev', 0, 0)",
            (PR_ID,),
        )
        yield conn


def test_candidates_are_saved_read_back_and_replaced(
    migrated: psycopg.Connection[TupleRow], tmp_path: Path
) -> None:
    """Check the file, the row's counts, a second save replacing the first, and other keys."""
    path = tmp_path / "k1" / f"{PR_ID}.json"
    save_context(migrated, "k1", path, candidates(piece("changed")))
    save_context(migrated, "k1", path, candidates(piece("changed"), piece("test")))
    assert stored_contexts(migrated, "k1") == {PR_ID: path}
    assert stored_contexts(migrated, "k2") == {}
    assert load_context(migrated, "k1", PR_ID).counts()["test"] == 1
    row = migrated.execute("SELECT pieces, changed_functions, stats FROM pr_contexts").fetchone()
    assert row == (2, 0, {"calls": 1})


def test_a_missing_file_counts_as_not_stored(
    migrated: psycopg.Connection[TupleRow], tmp_path: Path
) -> None:
    """Check a row whose file was deleted is retrieved again rather than failing later."""
    path = tmp_path / f"{PR_ID}.json"
    save_context(migrated, "k1", path, candidates(piece()))
    path.unlink()
    assert stored_contexts(migrated, "k1") == {}
    with pytest.raises(KeyError):
        load_context(migrated, "k1", PR_ID)
