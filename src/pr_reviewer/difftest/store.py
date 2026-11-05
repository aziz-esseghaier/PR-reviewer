"""Differential runs in PostgreSQL (table diff_runs, migration 0007), and reusing them.

Each run is kept under its request key (runner.request_key()): a fingerprint of the images,
the test code, the targets and the settings. diff_with_store() looks the key up first and
runs only when nothing is stored, so asking again for the same test on the same environment
costs nothing, like the LLM response cache. A stored run is replaced when it's run again on
purpose (`again`).

Imported by other modules; the file isn't run directly.
"""

import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.difftest.runner import (
    RUNS_PER_SIDE,
    DiffResult,
    FileSpec,
    SideSpec,
    request_key,
    run_diff,
)
from pr_reviewer.setup_agent.attempt import TryContext


def save_diff(conn: psycopg.Connection[TupleRow], result: DiffResult, runs: int) -> None:
    """Keep a differential run, replacing any stored under the same key.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection. bench_prs must hold the PR.
        result (DiffResult): The run.
        runs (int): Runs per side.
    """
    counts = result.counts()
    conn.execute(
        """
        INSERT INTO diff_runs (request_key, instance_id, old_side, new_side, runs_per_side,
                               tests, pass_fail, pass_pass, fail_pass, fail_fail, flaky,
                               not_run, minutes, details)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (request_key) DO UPDATE SET
            tests = EXCLUDED.tests, pass_fail = EXCLUDED.pass_fail,
            pass_pass = EXCLUDED.pass_pass, fail_pass = EXCLUDED.fail_pass,
            fail_fail = EXCLUDED.fail_fail, flaky = EXCLUDED.flaky, not_run = EXCLUDED.not_run,
            minutes = EXCLUDED.minutes, details = EXCLUDED.details, created_at = now()
        """,
        (
            result.key,
            result.instance_id,
            result.old_side,
            result.new_side,
            runs,
            len(result.tests),
            counts["pass_fail"],
            counts["pass_pass"],
            counts["fail_pass"],
            counts["fail_fail"],
            counts["flaky"],
            counts["not_run"],
            result.minutes,
            Jsonb(result.model_dump(mode="json")),
        ),
    )


def find_diff(conn: psycopg.Connection[TupleRow], key: str) -> DiffResult | None:
    """Look up a stored differential run.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        key (str): Its request key.

    Returns:
        DiffResult | None: The run, marked as reused, or None when none is stored.
    """
    row = conn.execute("SELECT details FROM diff_runs WHERE request_key = %s", (key,)).fetchone()
    if row is None:
        return None
    return DiffResult.model_validate({**row[0], "reused": True})


def diff_with_store(
    conn: psycopg.Connection[TupleRow],
    ctx: TryContext,
    pr: Instance,
    recipe: EnvRecipe,
    old: SideSpec,
    new: SideSpec,
    files: tuple[FileSpec, ...],
    targets: tuple[str, ...],
    *,
    runs: int = RUNS_PER_SIDE,
    timeout: float | None = None,
    again: bool = False,
) -> DiffResult:
    """Give a differential run's result: the stored one, or a new run that's then stored.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        ctx (TryContext): Docker, the sandbox settings and the log folder.
        pr (Instance): The PR.
        recipe (EnvRecipe): Its recipe.
        old (SideSpec): The side before.
        new (SideSpec): The side after.
        files (tuple[FileSpec, ...]): Test files written at both sides.
        targets (tuple[str, ...]): What to run.
        runs (int): Runs per side.
        timeout (float | None): Seconds per run. None means the config's difftest limit.
        again (bool): Run even when a result is stored, and replace it.

    Returns:
        DiffResult: Each test's outcome.
    """
    seconds = timeout if timeout is not None else ctx.config.timeouts.difftest_minutes * 60
    key = request_key(ctx, pr, recipe, (old, new), files, targets, runs, seconds)
    if not again:
        found = find_diff(conn, key)
        if found is not None:
            return found
    result = run_diff(
        ctx, pr, recipe, old, new, files, targets, runs=runs, timeout=seconds, key=key
    )
    save_diff(conn, result, runs)
    return result
