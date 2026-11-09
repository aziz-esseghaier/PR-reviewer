"""Static tool results in PostgreSQL (migration 0010) and on disk.

Each PR's full result is a JSON file under data/static/ (every warning on its changed files).
pr_static keeps one row per PR and tools setup, with the file's path and the counts, and
static_warnings one row per warning on a changed line, so warnings can be counted by tool,
rule or file with SQL. A PR whose tools couldn't run gets a row too, marked failed, so the
reviewer knows the tools are missing instead of thinking they found nothing.

Imported by other modules; the file isn't run directly.
"""

from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from pr_reviewer.core.config import project_root
from pr_reviewer.static.tools import StaticResult, load_result, save_result


@dataclass(frozen=True)
class StoredStatic:
    """One PR's stored static run.

    Attributes:
        instance_id (str): The PR.
        status (str): "ok", "no_python" (no Python line changed) or "failed".
        path (Path | None): The result file, when the tools ran.
        detail (str): Why it failed; empty otherwise.
    """

    instance_id: str
    status: str
    path: Path | None
    detail: str

    def load(self) -> StaticResult | None:
        """Read the full result.

        Returns:
            StaticResult | None: The result, or None for a failed run.
        """
        return load_result(self.path) if self.path is not None else None


def _relative(path: Path) -> str:
    """Write a path relative to the project folder when it's inside it.

    Args:
        path (Path): The path.

    Returns:
        str: The relative path, with "/", or the absolute one.
    """
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def save_static(
    conn: psycopg.Connection[TupleRow], key: str, path: Path, result: StaticResult
) -> None:
    """Write a PR's static result to its file and record it, replacing an earlier record.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        key (str): The static settings key.
        path (Path): The file to write.
        result (StaticResult): The result.
    """
    save_result(path, result)
    status = "ok" if result.changed else "no_python"
    with conn.transaction(), conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM pr_static WHERE instance_id = %s AND settings_key = %s",
            (result.instance_id, key),
        )
        cursor.execute(
            """
            INSERT INTO pr_static (instance_id, settings_key, status, head_commit, image, path,
                                   changed_files, warnings, kept, seconds, runs, detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, '')
            """,
            (
                result.instance_id,
                key,
                status,
                result.head,
                result.image,
                _relative(path),
                len(result.changed),
                len(result.warnings),
                len(result.kept),
                result.seconds,
                Jsonb([run.model_dump(mode="json") for run in result.runs]),
            ),
        )
        cursor.executemany(
            """
            INSERT INTO static_warnings (instance_id, settings_key, tool, rule, file, line,
                                         end_line, severity, message)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    result.instance_id,
                    key,
                    w.tool,
                    w.rule,
                    w.file,
                    w.line,
                    w.end_line,
                    w.severity,
                    w.message,
                )
                for w in result.kept
            ],
        )


def save_failure(
    conn: psycopg.Connection[TupleRow], key: str, instance_id: str, detail: str
) -> None:
    """Record that the tools couldn't run on a PR, replacing an earlier record.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        key (str): The static settings key.
        instance_id (str): The PR.
        detail (str): Why.
    """
    with conn.transaction(), conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM pr_static WHERE instance_id = %s AND settings_key = %s",
            (instance_id, key),
        )
        cursor.execute(
            """
            INSERT INTO pr_static (instance_id, settings_key, status, changed_files, warnings,
                                   kept, seconds, runs, detail)
            VALUES (%s, %s, 'failed', 0, 0, 0, 0, '[]', %s)
            """,
            (instance_id, key, detail[:2000]),
        )


def stored_static(conn: psycopg.Connection[TupleRow], key: str) -> dict[str, StoredStatic]:
    """List the PRs with a stored static run for a tools setup.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        key (str): The static settings key.

    Returns:
        dict[str, StoredStatic]: By PR. Runs whose file has gone missing are left out, so
        they're run again.
    """
    rows = conn.execute(
        "SELECT instance_id, status, path, detail FROM pr_static WHERE settings_key = %s",
        (key,),
    ).fetchall()
    found: dict[str, StoredStatic] = {}
    for instance_id, status, path, detail in rows:
        full = None
        if path is not None:
            full = Path(path) if Path(path).is_absolute() else project_root() / path
            if not full.is_file():
                continue
        found[instance_id] = StoredStatic(instance_id, status, full, detail)
    return found
