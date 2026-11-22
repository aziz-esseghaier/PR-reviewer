"""Carry a repo's working environment recipes from one Action run to the next.

The Action's database lives for one job, so the recipe cache (env_recipes) would start empty
every time, and the setup agent would spend model calls rebuilding what worked before. The
Action saves the repo's recipes to a JSON file in GitHub's cache after a run, and loads them
before the next one. A recipe only lists install steps and a test command, which run inside
sandboxes like the agent's own proposals; each one is checked against the recipe schema when
loaded, and recipes for another repo are refused.

GitHub scopes caches by branch: a run on a pull request reads caches saved on that pull
request or on the target branch, never another pull request's.

Imported by other modules; the file isn't run directly.
"""

import json
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import TupleRow
from pydantic import ValidationError

from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.setup_agent.store import save_recipe

# Most recipes kept per repo in the file.
MAX_RECIPES = 20


class RecipeFileError(Exception):
    """Raised when a recipe file isn't valid."""


def export_recipes(conn: psycopg.Connection[TupleRow], repo: str) -> list[dict[str, Any]]:
    """Read a repo's working recipes, newest first.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        repo (str): The repo, such as "octo/demo".

    Returns:
        list[dict[str, Any]]: Each recipe with its year and source.
    """
    rows = conn.execute(
        """
        SELECT year, source, recipe FROM env_recipes WHERE repo = %s
        ORDER BY created_at DESC LIMIT %s
        """,
        (repo, MAX_RECIPES),
    ).fetchall()
    return [{"year": int(y), "source": str(s), "recipe": r} for y, s, r in rows]


def read_recipes(path: Path, repo: str) -> list[tuple[int, str, EnvRecipe]]:
    """Read and check a recipe file.

    Args:
        path (Path): The file.
        repo (str): The repo the recipes must be for.

    Returns:
        list[tuple[int, str, EnvRecipe]]: Each recipe's year, source and the recipe.

    Raises:
        RecipeFileError: If the file isn't valid JSON of the expected shape, or holds a
            recipe for another repo.
    """
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise ValueError("not a list")
        recipes = []
        for item in items[:MAX_RECIPES]:
            year, source = int(item["year"]), str(item["source"])
            if source not in ("standard", "agent"):
                raise ValueError(f"unknown source {source!r}")
            recipe = EnvRecipe.model_validate(item["recipe"])
            if recipe.repo != repo:
                raise ValueError(f"a recipe for {recipe.repo}, not {repo}")
            recipes.append((year, source, recipe))
    except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
        raise RecipeFileError(f"{path.name} isn't a valid recipe file: {exc}") from None
    return recipes


def import_recipes(conn: psycopg.Connection[TupleRow], path: Path, repo: str) -> int:
    """Load a recipe file into the recipe cache.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        path (Path): The file. A missing file loads nothing.
        repo (str): The repo the recipes must be for.

    Returns:
        int: How many recipes were loaded.
    """
    if not path.is_file():
        return 0
    recipes = read_recipes(path, repo)
    for year, source, recipe in reversed(recipes):
        # The environment images they were saved with are gone; the setup rebuilds them.
        save_recipe(conn, recipe, year, source, None)
    return len(recipes)


def write_recipes(conn: psycopg.Connection[TupleRow], path: Path, repo: str) -> int:
    """Save a repo's working recipes to a file.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        path (Path): The file to write.
        repo (str): The repo.

    Returns:
        int: How many recipes were written.
    """
    recipes = export_recipes(conn, repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipes, indent=2) + "\n", encoding="utf-8", newline="\n")
    return len(recipes)
