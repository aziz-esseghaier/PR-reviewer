"""
Tests for keeping scores in benchmark/results.py: the results folder and the database tables.

Scores two small PRs with a scripted fake judge, then checks the folder's files (valid JSON
without NaN, a readable summary, the raw outputs), the rules for experiment names and reused
folders, and, in the tests marked `integration`, the rows written to PostgreSQL. Those use a
throwaway database and are skipped when Postgres isn't running.

How to run (from the project folder):
    pytest tests/benchmark/test_results.py
"""

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import psycopg
import pytest
from psycopg.rows import TupleRow

from pr_reviewer.benchmark.results import (
    REVIEW_CONFIG,
    Experiment,
    check_folder,
    plain_json,
    results_folder,
    save_bench_prs,
    save_scores,
    summary_markdown,
    write_results,
)
from pr_reviewer.benchmark.scoring import ScoreRun
from pr_reviewer.benchmark.split import Split
from pr_reviewer.core.config import ConfigError
from pr_reviewer.db.migrate import migrate
from tests.benchmark.test_judge import GOOD_CLEAN
from tests.benchmark.test_scoring import NO_POINTS, ONE_FALSE_ALARM, ONE_HIT, PRS, run_with

EXPERIMENT = Experiment(
    name="tiny",
    date="2026-10-30",
    split="dev",
    reviewer="none",
    predictions_file="reviews.jsonl",
    predictions_sha256="a" * 64,
    reviews=2,
    judge={"profile": "j", "provider": "p", "model": "m", "temperature": 0.0, "max_tokens": 8192},
)
SPLIT = Split(
    seed=0,
    dev_fraction=0.3,
    source_commit="0" * 40,
    dataset_sha256="0" * 64,
    dev=("psf__requests-1", "psf__requests-2"),
    test=(),
)


def scored(tmp_path: Path, run: ScoreRun) -> Path:
    """Write a run's results folder, with a reviews file to copy.

    Args:
        tmp_path (Path): Temporary folder for this test.
        run (ScoreRun): The scored run.

    Returns:
        Path: The results folder.
    """
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text('{"instance_id": "psf__requests-1", "review": "x"}\n', encoding="utf-8")
    folder = tmp_path / "results" / "tiny"
    write_results(folder, EXPERIMENT, run, reviews)
    return folder


def test_results_folder_holds_config_metrics_summary_and_raw(tmp_path: Path) -> None:
    """Check every file of a results folder, from a run of reviews with no points."""
    folder = scored(tmp_path, run_with([NO_POINTS, GOOD_CLEAN], "No issues.", "No issues."))
    assert json.loads((folder / "config.json").read_text())["predictions"]["sha256"] == "a" * 64
    metrics_text = (folder / "metrics.json").read_text()
    assert "NaN" not in metrics_text  # averages over no points are saved as null
    assert json.loads(metrics_text)["overall"]["avg_pred_severity"] is None
    summary = (folder / "summary.md").read_text()
    assert "| 0.500 | 0.000 | 0.000 | 0.000 | 0 | 1 | 0 | 1 |" in summary  # PR-level row
    assert "No review raised a point" in summary
    output = json.loads((folder / "raw" / "swrbench_output.json").read_text())
    assert [d["instance_id"] for d in output["details"]] == ["psf__requests-1", "psf__requests-2"]
    assert (folder / "raw" / "predictions.jsonl").is_file()


def test_summary_shows_functional_and_evolvability_rows() -> None:
    """Check the point-level table: all points, then F and E apart."""
    run = run_with([ONE_HIT, ONE_FALSE_ALARM], "1. Divides by zero.", "1. Odd spacing.")
    assert run.analysis is not None
    summary = summary_markdown(EXPERIMENT, plain_json(run.analysis), 2)
    assert "| All | 0.500 | 0.500 | 0.500 | 1 | 1 | 1 |" in summary
    assert "| Functional (F) | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |" in summary
    assert "| Evolvability (E) | 0.000 | 0.000 | 0.000 | 0 | 1 | 1 |" in summary


def test_numpy_numbers_and_nan_become_plain_json() -> None:
    """Check plain_json on the kinds of values SWR-Bench's metrics hold."""
    value = {"a": np.float64(0.25), "b": [np.int64(3), float("nan")], "c": (np.float64("nan"),)}
    assert plain_json(value) == {"a": 0.25, "b": [3, None], "c": [None]}
    assert type(plain_json(np.int64(3))) is int


@pytest.mark.parametrize("name", ["Baseline", "no comments", "../escape", "-dash", ""])
def test_bad_experiment_names_are_refused(name: str) -> None:
    """Check that names that make odd or unsafe folder names are refused."""
    with pytest.raises(ConfigError, match="Bad experiment name"):
        results_folder(name)


def test_folder_is_reused_only_for_the_same_reviews(tmp_path: Path) -> None:
    """Check that rescoring the same reviews file may reuse its folder, and nothing else may."""
    folder = scored(tmp_path, run_with([NO_POINTS, GOOD_CLEAN], "No issues.", "No issues."))
    check_folder(folder, "a" * 64)
    check_folder(tmp_path / "results" / "new", "b" * 64)
    with pytest.raises(ConfigError, match="already holds other results"):
        check_folder(folder, "b" * 64)
    other = tmp_path / "results" / "judge-choice"
    other.mkdir()
    (other / "decision.md").write_text("x")
    with pytest.raises(ConfigError, match="already holds other results"):
        check_folder(other, "a" * 64)


def test_a_review_runs_folder_takes_its_scores(tmp_path: Path) -> None:
    """Check a folder holding only a review run, not scored yet, is free for its scores."""
    folder = tmp_path / "results" / "naive"
    (folder / "raw").mkdir(parents=True)
    (folder / REVIEW_CONFIG).write_text("{}")
    (folder / "raw" / "reviews.jsonl").write_text("")
    check_folder(folder, "a" * 64)


def test_a_scored_subset_says_so() -> None:
    """Check `since` goes into config.json and the summary, and is absent otherwise."""
    subset = replace(EXPERIMENT, since=2022)
    assert subset.config()["since"] == 2022
    assert "since" not in EXPERIMENT.config()
    run = run_with([NO_POINTS, GOOD_CLEAN], "No issues.", "No issues.")
    assert run.analysis is not None
    summary = summary_markdown(subset, plain_json(run.analysis), 2)
    assert "2 PRs of the dev set from 2022 on (" in summary


@pytest.fixture
def migrated(test_db_url: str) -> Iterator[psycopg.Connection[TupleRow]]:
    """Open a connection to a fresh test database with all migrations applied.

    Args:
        test_db_url (str): Connection string of the empty throwaway database.

    Yields:
        psycopg.Connection[TupleRow]: Autocommit connection to the migrated database.
    """
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        migrate(conn)
        yield conn


@pytest.mark.integration
def test_scores_are_saved_per_pr_and_replaced_on_rescoring(
    migrated: psycopg.Connection[TupleRow],
) -> None:
    """Check bench_prs, experiments and pr_scores, and that a second save replaces the first."""
    save_bench_prs(migrated, PRS, SPLIT)
    run = run_with([ONE_HIT, ONE_FALSE_ALARM], "1. Divides by zero.", "1. Odd spacing.")
    save_scores(migrated, EXPERIMENT, run)
    save_scores(migrated, EXPERIMENT, run)
    rows = migrated.execute(
        """
        SELECT p.kind, s.judged_good, s.hits, s.false_alarms, s.misses,
               jsonb_array_length(s.judgment -> 'pred_points')
        FROM pr_scores s JOIN bench_prs p USING (instance_id)
        WHERE s.experiment = 'tiny' ORDER BY s.instance_id
        """
    ).fetchall()
    assert rows == [("functional", False, 1, 0, 1, 1), ("clean", False, 0, 1, 0, 1)]
    experiments = migrated.execute("SELECT name, reviewer, judge_model FROM experiments").fetchall()
    assert experiments == [("tiny", "none", "m")]
    issues = migrated.execute(
        "SELECT functional_issues, evolvability_issues, split FROM bench_prs ORDER BY 1 DESC"
    ).fetchall()
    assert issues == [(1, 1, "dev"), (0, 0, "dev")]
