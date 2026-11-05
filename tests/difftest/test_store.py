"""
Tests for keeping and reusing differential runs (difftest/store.py), and for finding a PR's
environment (difftest/environments.py).

They use a throwaway database and are skipped when Postgres isn't running.

How to run (from the project folder):
    pytest tests/difftest/test_store.py
"""

from collections.abc import Iterator
from dataclasses import replace
from typing import Any, cast

import psycopg
import pytest
from psycopg.rows import TupleRow

from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.db.migrate import migrate
from pr_reviewer.difftest import store
from pr_reviewer.difftest.environments import PrepareError, pr_environment
from pr_reviewer.difftest.runner import DiffResult, FileSpec, SideSpec
from pr_reviewer.setup_agent.attempt import TryContext
from pr_reviewer.setup_agent.store import save_recipe, save_run
from tests.difftest.helpers import make_pr, make_result
from tests.setup_agent.test_store import run

pytestmark = pytest.mark.integration

RECIPE = EnvRecipe(
    repo="psf/requests",
    python_version="3.9",
    install_steps=("uv pip install -e .",),
    test_command="python -m pytest",
)


@pytest.fixture
def db(test_db_url: str) -> Iterator[psycopg.Connection[TupleRow]]:
    """Give a migrated throwaway database with one benchmark PR.

    Args:
        test_db_url (str): Connection string of the empty test database.

    Yields:
        psycopg.Connection[TupleRow]: Autocommit connection to it.
    """
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        migrate(conn)
        conn.execute(
            "INSERT INTO bench_prs VALUES ('psf__requests-1', 'psf/requests', 2022, 'clean', "
            "'dev', 0, 0)"
        )
        yield conn


def test_a_stored_run_comes_back_marked_as_reused(db: psycopg.Connection[TupleRow]) -> None:
    """Check a run is kept by its key, comes back whole, and a second save replaces it."""
    first = make_result({"t::a": "pass_fail", "t::b": "flaky"}, key="key-1")
    store.save_diff(db, first, 3)
    found = store.find_diff(db, "key-1")
    assert found is not None and found.reused
    assert found.model_copy(update={"reused": False}) == first
    store.save_diff(db, make_result({"t::a": "pass_pass"}, key="key-1"), 3)
    row = db.execute(
        "SELECT count(*), max(tests), max(pass_fail), max(pass_pass) FROM diff_runs"
    ).fetchone()
    assert row == (1, 1, 0, 1)
    assert store.find_diff(db, "missing") is None


def test_the_same_request_runs_once(
    db: psycopg.Connection[TupleRow], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check diff_with_store runs once, then reuses, and runs again when asked."""
    calls: list[str] = []

    def fake_run(*args: Any, key: str, **kwargs: Any) -> DiffResult:
        calls.append(key)
        return make_result({"t::a": "pass_fail"}, key=key)

    monkeypatch.setattr(store, "request_key", lambda *args: "key-2")
    monkeypatch.setattr(store, "run_diff", fake_run)
    ctx = cast(TryContext, None)
    sides = (SideSpec("base", "a"), SideSpec("pr", "b"))
    files = (FileSpec("t.py", ""),)

    def ask(again: bool = False) -> DiffResult:
        return store.diff_with_store(
            db, ctx, make_pr(), RECIPE, *sides, files, ("t.py",), timeout=60, again=again
        )

    assert not ask().reused
    assert ask().reused
    assert not ask(again=True).reused
    assert calls == ["key-2", "key-2"]


def test_a_prs_environment_is_the_recipe_of_its_latest_passing_setup(
    db: psycopg.Connection[TupleRow],
) -> None:
    """Check the recipe and image come from the latest setup, and a failed one has none."""
    pr = make_pr()
    with pytest.raises(PrepareError, match="no environment yet"):
        pr_environment(db, pr)
    recipe_id = save_recipe(db, RECIPE, 2022, "standard", "pr-reviewer-env:x")
    save_run(db, replace(run("psf__requests-1", True), recipe_id=recipe_id))
    env = pr_environment(db, pr)
    assert (env.recipe, env.start_image) == (RECIPE, "pr-reviewer-env:x")
    save_run(db, run("psf__requests-1", False, "build_failed"))
    with pytest.raises(PrepareError, match="didn't pass"):
        pr_environment(db, pr)
