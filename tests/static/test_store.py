"""Tests for keeping static results (static/store.py): files on disk, rows in PostgreSQL.

Checks that a result is saved as a file and recorded under its settings key with one row per
warning on a changed line, that saving again replaces it, that a failed run is recorded
without a file, and that a result whose file is gone counts as not stored. Marked
`integration`: they use a throwaway database and are skipped when Postgres isn't running.

How to run (from the project folder):
    pytest tests/static/test_store.py
"""

from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import TupleRow

from pr_reviewer.db.migrate import migrate
from pr_reviewer.static.store import save_failure, save_static, stored_static
from pr_reviewer.static.tools import StaticResult, ToolRun
from tests.static.test_warnings import warning

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


def result(kept: int) -> StaticResult:
    """Make a result with some warnings on changed lines.

    Args:
        kept (int): How many warnings touch a changed line.

    Returns:
        StaticResult: The result.
    """
    on_lines = [warning(line=line) for line in range(1, kept + 1)]
    return StaticResult(
        instance_id=PR_ID,
        version=1,
        image="pr-reviewer-pr:psf__requests-1-pr",
        base="a" * 40,
        head="b" * 40,
        changed={"pkg/cart.py": list(range(1, 10))},
        runs=[ToolRun(tool="mypy", status="ok", seconds=1.0, mode="silent", warnings=kept + 1)],
        warnings=[*on_lines, warning(line=50)],
        kept=on_lines,
        seconds=5.0,
    )


def test_a_result_is_saved_counted_and_replaced(
    migrated: psycopg.Connection[TupleRow], tmp_path: Path
) -> None:
    """Check the file, the row's counts, the warning rows, and a second save replacing them."""
    path = tmp_path / "k1" / f"{PR_ID}.json"
    save_static(migrated, "k1", path, result(kept=2))
    save_static(migrated, "k1", path, result(kept=1))
    stored = stored_static(migrated, "k1")
    assert list(stored) == [PR_ID] and stored[PR_ID].status == "ok"
    loaded = stored[PR_ID].load()
    assert loaded is not None and len(loaded.kept) == 1
    assert stored_static(migrated, "k2") == {}
    row = migrated.execute("SELECT changed_files, warnings, kept FROM pr_static").fetchone()
    assert row == (1, 2, 1)
    rows = migrated.execute("SELECT tool, rule, line FROM static_warnings").fetchall()
    assert rows == [("mypy", "attr-defined", 1)]


def test_a_failed_run_is_recorded_without_a_file(migrated: psycopg.Connection[TupleRow]) -> None:
    """Check a failure is stored with its reason and no result to load."""
    save_failure(migrated, "k1", PR_ID, "installing at the pr side failed")
    stored = stored_static(migrated, "k1")[PR_ID]
    assert stored.status == "failed" and stored.path is None and stored.load() is None
    assert stored.detail == "installing at the pr side failed"


def test_a_missing_file_counts_as_not_stored(
    migrated: psycopg.Connection[TupleRow], tmp_path: Path
) -> None:
    """Check a row whose file was deleted is run again rather than failing later."""
    path = tmp_path / f"{PR_ID}.json"
    save_static(migrated, "k1", path, result(kept=1))
    path.unlink()
    assert stored_static(migrated, "k1") == {}
