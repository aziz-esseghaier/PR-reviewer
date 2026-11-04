"""The recipe cache and the setup runs, in PostgreSQL (tables from migration 0006).

- save_recipe() keeps a recipe that built a working environment, per repo and year, with the
  environment image saved from it.
- cached_recipes() gives the recipes to try first for a PR: its own year's, then the nearest
  years', newest first. So a repo's second PR of a year usually needs no model call at all.
- save_run() records one setup of a PR, and latest_runs() gives each PR's most recent one, for
  the report and the usable subset.

Imported by other modules; the file isn't run directly.
"""

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.setup_agent.attempt import recipe_fingerprint

# How many cached recipes a setup tries before the standard recipe.
CACHED_TRIES = 2


@dataclass(frozen=True)
class StoredRecipe:
    """A recipe from the cache.

    Attributes:
        id (int): Its row ID.
        year (int): The year it was made for.
        source (str): Where it first came from: "standard" or "agent".
        recipe (EnvRecipe): The recipe.
        image (str | None): The environment image built from it, if any.
    """

    id: int
    year: int
    source: str
    recipe: EnvRecipe
    image: str | None


@dataclass(frozen=True)
class SetupRun:
    """One finished setup of a PR, as saved in setup_runs.

    Attributes:
        instance_id (str): The PR.
        thread_id (str): The setup graph's thread.
        repo (str): Its repo.
        year (int): Its year.
        verified (bool): Whether an environment passed at both sides.
        source (str | None): What got it working: "cache", "standard" or "agent".
        failure_category (str | None): Why it failed, when it did.
        recipe_id (int | None): The recipe that worked.
        attempts (int): Recipes tried.
        model_calls (int): Calls to the setup agent's model.
        minutes (float): Time spent in attempts.
        base_pass_rate (float | None): Share of related tests passing at the base.
        pr_pass_rate (float | None): Share passing at the PR state.
        tests_run (int | None): Related tests that ran at the base.
        agent_model (str): The setup agent's model.
        details (dict[str, Any]): Every attempt and the related test files.
    """

    instance_id: str
    thread_id: str
    repo: str
    year: int
    verified: bool
    source: str | None
    failure_category: str | None
    recipe_id: int | None
    attempts: int
    model_calls: int
    minutes: float
    base_pass_rate: float | None
    pr_pass_rate: float | None
    tests_run: int | None
    agent_model: str
    details: dict[str, Any]


def save_recipe(
    conn: psycopg.Connection[TupleRow],
    recipe: EnvRecipe,
    year: int,
    source: str,
    image: str | None,
) -> int:
    """Keep a recipe that worked, or update its image if it's already kept.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        recipe (EnvRecipe): The recipe.
        year (int): The year of the PR it worked for.
        source (str): "standard" or "agent".
        image (str | None): The environment image saved from it.

    Returns:
        int: The recipe's row ID.
    """
    row = conn.execute(
        """
        INSERT INTO env_recipes (repo, year, fingerprint, source, recipe, image)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (repo, year, fingerprint) DO UPDATE
            SET image = COALESCE(EXCLUDED.image, env_recipes.image)
        RETURNING id
        """,
        (
            recipe.repo,
            year,
            recipe_fingerprint(recipe),
            source,
            Jsonb(recipe.model_dump(mode="json")),
            image,
        ),
    ).fetchone()
    assert row is not None
    return int(row[0])


def cached_recipes(
    conn: psycopg.Connection[TupleRow], repo: str, year: int, limit: int = CACHED_TRIES
) -> list[StoredRecipe]:
    """Give the kept recipes to try first for a PR: same year first, then the nearest years.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        repo (str): The PR's repo.
        year (int): The PR's year.
        limit (int): Most recipes to give.

    Returns:
        list[StoredRecipe]: The recipes, best first.
    """
    rows = conn.execute(
        """
        SELECT id, year, source, recipe, image FROM env_recipes
        WHERE repo = %s
        ORDER BY abs(year - %s), year DESC, created_at DESC
        LIMIT %s
        """,
        (repo, year, limit),
    ).fetchall()
    return [
        StoredRecipe(int(r[0]), int(r[1]), str(r[2]), EnvRecipe.model_validate(r[3]), r[4])
        for r in rows
    ]


def save_run(conn: psycopg.Connection[TupleRow], run: SetupRun) -> None:
    """Record one finished setup of a PR.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection. bench_prs must hold the PR.
        run (SetupRun): The setup.
    """
    conn.execute(
        """
        INSERT INTO setup_runs (instance_id, thread_id, repo, year, verified, source,
                                failure_category, recipe_id, attempts, model_calls, minutes,
                                base_pass_rate, pr_pass_rate, tests_run, agent_model, details)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            run.instance_id,
            run.thread_id,
            run.repo,
            run.year,
            run.verified,
            run.source,
            run.failure_category,
            run.recipe_id,
            run.attempts,
            run.model_calls,
            run.minutes,
            run.base_pass_rate,
            run.pr_pass_rate,
            run.tests_run,
            run.agent_model,
            Jsonb(run.details),
        ),
    )


def latest_runs(conn: psycopg.Connection[TupleRow]) -> dict[str, dict[str, Any]]:
    """Give each PR's most recent setup.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.

    Returns:
        dict[str, dict[str, Any]]: The setup_runs row of each PR, by instance ID.
    """
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        rows = cursor.execute(
            """
            SELECT DISTINCT ON (instance_id) *
            FROM setup_runs
            ORDER BY instance_id, finished_at DESC, id DESC
            """
        ).fetchall()
    return {row["instance_id"]: row for row in rows}
