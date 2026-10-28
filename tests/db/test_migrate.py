"""Tests for the migration runner in db/migrate.py.

Checks that migration files are found in number order and badly named ones are refused. The
tests marked `integration` apply migrations to a throwaway database: each runs once, pgvector
gets enabled, and a broken migration leaves nothing behind. They are skipped when Postgres
isn't running.

How to run (from the project folder):
    pytest tests/db/test_migrate.py                       all migration tests
    pytest tests/db/test_migrate.py -m "not integration"  without the database
"""

from pathlib import Path

import psycopg
import pytest

from pr_reviewer.db.connection import extension_version
from pr_reviewer.db.migrate import (
    MIGRATIONS_DIR,
    applied_migrations,
    discover_migrations,
    migrate,
)


def _write_migrations(directory: Path, *names: str) -> None:
    """Create small migration files with the given names.

    Args:
        directory (Path): Folder to create the files in.
        *names (str): File names to create.
    """
    for name in names:
        (directory / name).write_text("SELECT 1;\n", encoding="utf-8")


def test_discover_sorts_by_version(tmp_path: Path) -> None:
    """Check that migrations come back oldest first, whatever order the files were made in."""
    _write_migrations(tmp_path, "0002_second.sql", "0010_tenth.sql", "0001_first.sql")
    assert [m.version for m in discover_migrations(tmp_path)] == ["0001", "0002", "0010"]
    assert discover_migrations(tmp_path)[0].name == "first"


@pytest.mark.parametrize(
    "bad_name", ["1_short.sql", "0001-dash.sql", "0001_Upper.sql", "0001_.sql"]
)
def test_discover_rejects_bad_names(tmp_path: Path, bad_name: str) -> None:
    """Check that a badly named .sql file stops the run instead of being skipped silently."""
    _write_migrations(tmp_path, bad_name)
    with pytest.raises(ValueError, match="Bad migration file name"):
        discover_migrations(tmp_path)


def test_discover_rejects_duplicate_versions(tmp_path: Path) -> None:
    """Check that two files with the same number are refused."""
    _write_migrations(tmp_path, "0001_a.sql", "0001_b.sql")
    with pytest.raises(ValueError, match="share version 0001"):
        discover_migrations(tmp_path)


def test_project_migrations_are_valid() -> None:
    """Check that the migrations shipped with the package are well named and start at 0001."""
    migrations = discover_migrations(MIGRATIONS_DIR)
    assert migrations[0].version == "0001"


@pytest.mark.integration
def test_migrate_applies_each_migration_once(test_db_url: str) -> None:
    """Check that the first run applies every migration and a second run applies nothing."""
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        first_run = migrate(conn)
        second_run = migrate(conn)
        applied = applied_migrations(conn)
    assert first_run == discover_migrations()
    assert second_run == []
    assert set(applied) == {m.version for m in first_run}


@pytest.mark.integration
def test_migrate_enables_pgvector(test_db_url: str) -> None:
    """Check that pgvector is missing in a fresh database and installed after migrating."""
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        assert extension_version(conn, "vector") is None
        migrate(conn)
        assert extension_version(conn, "vector") is not None


@pytest.mark.integration
def test_failed_migration_is_rolled_back(test_db_url: str, tmp_path: Path) -> None:
    """Check that a broken migration leaves no changes behind and isn't recorded as applied."""
    (tmp_path / "0001_good.sql").write_text("CREATE TABLE good (id INT);", encoding="utf-8")
    (tmp_path / "0002_bad.sql").write_text(
        "CREATE TABLE half (id INT);\nTHIS IS NOT SQL;", encoding="utf-8"
    )
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        with pytest.raises(psycopg.errors.SyntaxError):
            migrate(conn, tmp_path)
        assert set(applied_migrations(conn)) == {"0001"}
        half_table = conn.execute("SELECT to_regclass('half')").fetchone()
    assert half_table == (None,)
