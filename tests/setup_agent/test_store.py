"""
Tests for the recipe cache and setup runs in setup_agent/store.py, and the setup report.

The store tests use a throwaway database and are skipped when Postgres isn't running; the
report test needs nothing.

How to run (from the project folder):
    pytest tests/setup_agent/test_store.py
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg.rows import TupleRow

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.db.migrate import migrate
from pr_reviewer.setup_agent.report import write_report
from pr_reviewer.setup_agent.store import (
    SetupRun,
    cached_recipes,
    latest_runs,
    save_recipe,
    save_run,
)
from tests.benchmark.test_dataset import make_row


def recipe(python: str = "3.9", repo: str = "psf/requests") -> EnvRecipe:
    """Make a recipe.

    Args:
        python (str): Its Python version.
        repo (str): Its repo.

    Returns:
        EnvRecipe: The recipe.
    """
    return EnvRecipe(
        repo=repo,
        python_version=python,
        install_steps=("uv pip install -e .",),
        test_command="python -m pytest",
    )


def run(instance_id: str, verified: bool, category: str | None = None) -> SetupRun:
    """Make a finished setup.

    Args:
        instance_id (str): The PR.
        verified (bool): Whether it worked.
        category (str | None): Why it failed.

    Returns:
        SetupRun: The setup.
    """
    return SetupRun(
        instance_id=instance_id,
        thread_id=f"setup-{instance_id}",
        repo="psf/requests",
        year=2022,
        verified=verified,
        source="standard" if verified else None,
        failure_category=category,
        recipe_id=None,
        attempts=1,
        model_calls=0 if verified else 3,
        minutes=4.5,
        base_pass_rate=0.95 if verified else None,
        pr_pass_rate=0.96 if verified else None,
        tests_run=40 if verified else None,
        agent_model="setup-gpt-oss-120b",
        details={"attempts": []},
    )


@pytest.fixture
def db(test_db_url: str) -> Iterator[psycopg.Connection[TupleRow]]:
    """Give a migrated throwaway database with two benchmark PRs.

    Args:
        test_db_url (str): Connection string of the empty test database.

    Yields:
        psycopg.Connection[TupleRow]: Autocommit connection to it.
    """
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        migrate(conn)
        for number in (1, 2):
            conn.execute(
                "INSERT INTO bench_prs VALUES (%s, 'psf/requests', 2022, 'clean', 'dev', 0, 0)",
                (f"psf__requests-{number}",),
            )
        yield conn


@pytest.mark.integration
def test_recipes_are_kept_once_and_found_by_nearest_year(db: psycopg.Connection[TupleRow]) -> None:
    """Check one row per recipe and year (a later image fills the gap), nearest year first."""
    first = save_recipe(db, recipe("3.9"), 2022, "standard", None)
    again = save_recipe(db, recipe("3.9"), 2022, "standard", "pr-reviewer-env:a")
    assert first == again
    save_recipe(db, recipe("3.8"), 2019, "agent", None)
    save_recipe(db, recipe("3.10", repo="other/repo"), 2023, "agent", None)
    found = cached_recipes(db, "psf/requests", 2021)
    assert [(r.year, r.recipe.python_version, r.source, r.image) for r in found] == [
        (2022, "3.9", "standard", "pr-reviewer-env:a"),
        (2019, "3.8", "agent", None),
    ]


@pytest.mark.integration
def test_latest_run_of_each_pr_wins(db: psycopg.Connection[TupleRow]) -> None:
    """Check the most recent setup of a PR replaces earlier ones in latest_runs()."""
    save_run(db, run("psf__requests-1", False, "build_failed"))
    save_run(db, run("psf__requests-1", True))
    save_run(db, run("psf__requests-2", False, "tests_failing"))
    latest = latest_runs(db)
    assert latest["psf__requests-1"]["verified"] is True
    assert latest["psf__requests-2"]["failure_category"] == "tests_failing"


def test_report_counts_rates_sources_failures_and_the_subset(tmp_path: Path) -> None:
    """Check the summary's rates by repo and year, its failure table, and usable_prs.json.

    A run of a PR out of scope (an older one, say) is left out of all of them.
    """
    prs = [
        Instance.model_validate_json(json.dumps(make_row(n, year=y, issues=())))
        for n, y in ((1, 2022), (2, 2022), (3, 2023))
    ]
    runs: dict[str, dict[str, Any]] = {
        "psf__requests-1": {**run("psf__requests-1", True).__dict__},
        "psf__requests-2": {**run("psf__requests-2", False, "build_failed").__dict__},
        "psf__requests-9": {**run("psf__requests-9", True).__dict__},
    }
    config = {
        "since": 2022,
        "agent_model": "setup-gpt-oss-120b",
        "budget": {"model_calls": 12, "minutes": 60.0},
        "min_pass_rate": 0.9,
    }
    usable = write_report(tmp_path, prs, runs, config)
    assert usable == ["psf__requests-1"]
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "- 2 of 3 dev PRs from 2022 onward set up so far" in summary
    assert "| psf/requests | 1/2 (50%) |" in summary
    assert "| 2022 | 1/2 (50%) |" in summary
    assert "| the standard recipe | 1 |" in summary
    assert "| build_failed | 1 |" in summary
    assert json.loads((tmp_path / "usable_prs.json").read_text(encoding="utf-8")) == usable
    assert len((tmp_path / "raw" / "runs.jsonl").read_text(encoding="utf-8").splitlines()) == 2
