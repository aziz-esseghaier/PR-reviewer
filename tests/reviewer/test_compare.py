"""Tests for comparing two scored runs (reviewer/compare.py).

Scores two runs of the same two PRs with SWR-Bench's scorer and a scripted fake judge, then
checks the comparison: SWR-Bench's metrics on all PRs and on the subsets with and without
context, and the PR-by-PR changes in hits and false alarms.

How to run (from the project folder):
    pytest tests/reviewer/test_compare.py
"""

import json
from pathlib import Path
from typing import Any

import pytest

from pr_reviewer.benchmark.source import DatasetError
from pr_reviewer.reviewer.compare import comparison_markdown, load_judgments, same_prs
from tests.benchmark.test_judge import GOOD_CLEAN
from tests.benchmark.test_scoring import NO_POINTS, ONE_FALSE_ALARM, ONE_HIT, PRS, run_with


def judgments(replies: list[str | int], *texts: str) -> dict[str, dict[str, Any]]:
    """Score reviews of the two test PRs and give the judgments by PR.

    Args:
        replies (list[str | int]): The fake judge's replies.
        *texts (str): The review texts.

    Returns:
        dict[str, dict[str, Any]]: Each PR's judgment.
    """
    run = run_with(replies, *texts)
    return {result["instance_id"]: result for result in run.results}


def test_the_comparison_shows_subsets_and_pr_changes() -> None:
    """Check rows for all PRs and both subsets, and the PR whose hit was gained."""
    base = judgments([NO_POINTS, GOOD_CLEAN], "Fine.", "Fine.")
    new = judgments([ONE_HIT, ONE_FALSE_ALARM], "1. Divides by zero.", "1. Odd spacing.")
    issues, clean = PRS[0].instance_id, PRS[1].instance_id
    subsets = [("with context", {issues}), ("without context", {clean})]
    text = comparison_markdown("naive", "context", base, new, subsets)
    assert "| All 2 | naive | 0.500 | 0.000 | 0.000 | 0.000 |" in text
    # It flags the PR with issues (right) and the clean one (wrong): accuracy 1 of 2.
    assert "| All 2 | context | 0.500 | 0.500 | 0.500 | 0.500 |" in text
    assert "| 1 with context | context |" in text
    assert "| 1 without context | naive |" in text
    assert "- Hits: 1 with context, 0 with naive; 1 PRs gained hits, 0 lost some" in text
    assert "False alarms on the 1 clean PRs: 1 (was 0)" in text
    assert f"| {issues} | 2 | 0 → 1 | 0 → 0 |" in text
    assert f"| {clean} | 0 | 0 → 0 | 0 → 1 |" in text


def test_runs_over_different_prs_are_refused() -> None:
    """Check two runs must have scored the same PRs."""
    base = judgments([NO_POINTS, GOOD_CLEAN], "Fine.", "Fine.")
    one = {PRS[0].instance_id: base[PRS[0].instance_id]}
    with pytest.raises(DatasetError, match="scored different PRs"):
        comparison_markdown("a", "b", base, one)


def test_a_run_over_some_prs_is_compared_on_those() -> None:
    """Check a run over the PRs with issues is compared with the base run on the same PRs."""
    base = judgments([NO_POINTS, GOOD_CLEAN], "Fine.", "Fine.")
    issues = PRS[0].instance_id
    both = judgments([ONE_HIT, GOOD_CLEAN], "1. Divides by zero.", "Fine.")
    new = {issues: both[issues]}
    narrowed = same_prs(base, new)
    assert list(narrowed) == [issues]
    text = comparison_markdown("context", "candidates", narrowed, new)
    # One hit of the PR's two issues, and no false alarm.
    assert "| All 1 | candidates | 1.000 | 1.000 | 0.500 | 0.667 |" in text
    assert "| All 1 | context | 0.000 | 0.000 | 0.000 | 0.000 |" in text
    # A run over a PR the base run didn't score can't be narrowed.
    assert same_prs(new, base) == new


def test_judgments_are_read_from_the_results_folder(tmp_path: Path) -> None:
    """Check the judgments come from raw/swrbench_output.json, and a missing one is named."""
    base = judgments([NO_POINTS, GOOD_CLEAN], "Fine.", "Fine.")
    (tmp_path / "raw").mkdir()
    output = {"analysis_results": {}, "details": list(base.values())}
    (tmp_path / "raw" / "swrbench_output.json").write_text(json.dumps(output), encoding="utf-8")
    assert load_judgments(tmp_path) == base
    with pytest.raises(DatasetError, match="isn't scored yet"):
        load_judgments(tmp_path / "missing")
