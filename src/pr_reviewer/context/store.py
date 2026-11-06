"""Retrieved context in PostgreSQL (migration 0009) and on disk.

Each PR's candidates are a JSON file under data/contexts/ (large text stays on disk), and
pr_contexts keeps one row per PR and retrieval setup: the file's path and the counts. A PR is
retrieved once per setup; reviews and the retrieval check read the stored candidates.

Imported by other modules; the file isn't run directly.
"""

from pathlib import Path

import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from pr_reviewer.context.candidates import Candidates, load_candidates, save_candidates
from pr_reviewer.core.config import project_root


def save_context(
    conn: psycopg.Connection[TupleRow], key: str, path: Path, candidates: Candidates
) -> None:
    """Write a PR's candidates to their file and record them, replacing an earlier record.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        key (str): The retrieval settings key.
        path (Path): The file to write.
        candidates (Candidates): The candidates.
    """
    save_candidates(path, candidates)
    try:
        relative = path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        relative = path.resolve().as_posix()
    conn.execute(
        """
        INSERT INTO pr_contexts (instance_id, settings_key, probe_version, head_commit, path,
                                 changed_functions, pieces, seconds, stats)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (instance_id, settings_key) DO UPDATE SET
            probe_version = EXCLUDED.probe_version, head_commit = EXCLUDED.head_commit,
            path = EXCLUDED.path, changed_functions = EXCLUDED.changed_functions,
            pieces = EXCLUDED.pieces, seconds = EXCLUDED.seconds, stats = EXCLUDED.stats,
            created_at = now()
        """,
        (
            candidates.instance_id,
            key,
            candidates.version,
            candidates.head,
            relative,
            len(candidates.changed),
            len(candidates.pieces),
            candidates.seconds,
            Jsonb(candidates.stats),
        ),
    )


def stored_contexts(conn: psycopg.Connection[TupleRow], key: str) -> dict[str, Path]:
    """List the PRs whose candidates are stored for a retrieval setup.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        key (str): The retrieval settings key.

    Returns:
        dict[str, Path]: The candidates file, by PR. PRs whose file has gone missing are left
        out, so they're retrieved again.
    """
    rows = conn.execute(
        "SELECT instance_id, path FROM pr_contexts WHERE settings_key = %s", (key,)
    ).fetchall()
    found: dict[str, Path] = {}
    for instance_id, path in rows:
        full = Path(path) if Path(path).is_absolute() else project_root() / path
        if full.is_file():
            found[instance_id] = full
    return found


def load_context(conn: psycopg.Connection[TupleRow], key: str, instance_id: str) -> Candidates:
    """Read a PR's stored candidates.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        key (str): The retrieval settings key.
        instance_id (str): The PR.

    Returns:
        Candidates: The candidates.

    Raises:
        KeyError: If none are stored for this setup.
    """
    path = stored_contexts(conn, key).get(instance_id)
    if path is None:
        raise KeyError(instance_id)
    return load_candidates(path)
