"""Migration runner: brings the database tables up to date.

A migration is a numbered SQL file in db/migrations/, such as 0002_llm_cache.sql. This module
finds those files, applies the ones the database doesn't have yet in number order, and records
each one in the schema_migrations table so it never runs twice.

How to run (from the project folder, with the database up):
    pr-reviewer db migrate     apply new migrations
    pr-reviewer db status      show which migrations are applied
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg.rows import TupleRow

from pr_reviewer.db.connection import table_exists

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
# Four-digit version, then a lowercase name: 0001_enable_pgvector.sql
FILENAME_PATTERN = re.compile(r"^(\d{4})_([a-z0-9]+(?:_[a-z0-9]+)*)\.sql$")


@dataclass(frozen=True)
class Migration:
    """One SQL migration file.

    Attributes:
        version (str): Four-digit number that sets the order, such as "0001".
        name (str): Short description from the file name, such as "enable_pgvector".
        path (Path): Location of the .sql file.
    """

    version: str
    name: str
    path: Path


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Find the migration files in a folder and sort them by version.

    Args:
        directory (Path): Folder that holds the .sql files.

    Returns:
        list[Migration]: The migrations, oldest first.

    Raises:
        ValueError: If a .sql file is badly named or two files share a version.
    """
    found: dict[str, Migration] = {}
    for path in directory.glob("*.sql"):
        match = FILENAME_PATTERN.match(path.name)
        if match is None:
            raise ValueError(
                f"Bad migration file name {path.name!r}: expected a name like 0002_llm_cache.sql"
            )
        version, name = match.groups()
        if version in found:
            other = found[version].path.name
            raise ValueError(f"Two migrations share version {version}: {other} and {path.name}")
        found[version] = Migration(version=version, name=name, path=path)
    return [found[version] for version in sorted(found)]


def ensure_migrations_table(conn: psycopg.Connection[TupleRow]) -> None:
    """Create the table that records applied migrations, if it doesn't exist yet.

    Args:
        conn (psycopg.Connection[TupleRow]): Open database connection.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def applied_migrations(conn: psycopg.Connection[TupleRow]) -> dict[str, datetime]:
    """Read which migrations the database has already applied, without changing anything.

    Args:
        conn (psycopg.Connection[TupleRow]): Open database connection.

    Returns:
        dict[str, datetime]: Applied versions mapped to the time they were applied.
            Empty if the database has never been migrated.
    """
    if not table_exists(conn, "schema_migrations"):
        return {}
    rows = conn.execute("SELECT version, applied_at FROM schema_migrations").fetchall()
    return {version: applied_at for version, applied_at in rows}


def migrate(
    conn: psycopg.Connection[TupleRow], directory: Path = MIGRATIONS_DIR
) -> list[Migration]:
    """Apply every migration the database doesn't have yet, oldest first.

    Each file runs in its own transaction together with its row in schema_migrations.
    If a file fails, its changes are rolled back and it isn't recorded, so the next run
    tries it again.

    Args:
        conn (psycopg.Connection[TupleRow]): Open connection in autocommit mode.
        directory (Path): Folder that holds the .sql files.

    Returns:
        list[Migration]: The migrations this call applied. Empty if the database was up to date.
    """
    ensure_migrations_table(conn)
    already_applied = applied_migrations(conn)
    newly_applied: list[Migration] = []
    for migration in discover_migrations(directory):
        if migration.version in already_applied:
            continue
        with conn.transaction():
            conn.execute(migration.path.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                (migration.version, migration.name),
            )
        newly_applied.append(migration)
    return newly_applied
