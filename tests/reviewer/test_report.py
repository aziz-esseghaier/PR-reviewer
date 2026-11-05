"""Tests for writing a review run's results folder (reviewer/report.py).

Checks that the settings and summary are always written, that the reviews file waits until
every PR has a review, and that once written it holds one review per PR in the run's order,
with the findings next to it.

How to run (from the project folder):
    pytest tests/reviewer/test_report.py
"""

import json
from pathlib import Path
from typing import Any

from pr_reviewer.benchmark.results import REVIEW_CONFIG
from pr_reviewer.benchmark.scoring import load_predictions
from pr_reviewer.reviewer.report import REVIEW_SUMMARY, write_review_files
from pr_reviewer.reviewer.store import SavedReview
from tests.reviewer.helpers import finding, make_pr

PRS = [make_pr(number=1), make_pr(number=2)]
SETTINGS: dict[str, Any] = {
    "name": "naive-test",
    "reviewer": "naive",
    "split": "dev",
    "since": 2022,
    "profile": "reviewer-x",
    "model": "x",
    "temperature": 0.0,
    "max_tokens": 3000,
    "reasoning_effort": "medium",
    "input_budget_chars": 12000,
    "max_tries": 3,
}


def saved(pr_number: int, status: str = "done", findings: int = 1) -> SavedReview:
    """Make a saved review of one of the test PRs.

    Args:
        pr_number (int): 1 or 2.
        status (str): "done" or "failed".
        findings (int): How many findings it has.

    Returns:
        SavedReview: The review.
    """
    return SavedReview(
        instance_id=PRS[pr_number - 1].instance_id,
        status=status,
        failures=0 if status == "done" else 1,
        tries=1,
        findings=findings,
        trimmed=pr_number == 2,
        input_tokens=1000,
        output_tokens=500,
        review=f"1. review of PR {pr_number}" if status == "done" else "",
        left_out=("pkg/b.py (commit 1234567)",) if pr_number == 2 else (),
    )


FINDINGS = [
    {"instance_id": PRS[0].instance_id, "id": "f1", "claim": finding(), "evidence": []},
    {
        "instance_id": PRS[1].instance_id,
        "id": "f1",
        "claim": finding(category="E.1.1"),
        "evidence": [],
    },
]


def test_the_reviews_file_waits_for_every_pr(tmp_path: Path) -> None:
    """Check a run with a failed PR writes its settings and summary, but no reviews file."""
    reviews = {r.instance_id: r for r in (saved(1), saved(2, status="failed"))}
    assert write_review_files(tmp_path, SETTINGS, PRS, reviews, FINDINGS) is None
    assert json.loads((tmp_path / REVIEW_CONFIG).read_text(encoding="utf-8")) == SETTINGS
    summary = (tmp_path / REVIEW_SUMMARY).read_text(encoding="utf-8")
    assert "Reviewed: 1; failed (no usable reply in 3 tries): 1; not reviewed yet: 0" in summary
    assert f"Failed PRs: {PRS[1].instance_id}" in summary
    assert not (tmp_path / "raw").exists()


def test_a_finished_run_writes_one_review_per_pr(tmp_path: Path) -> None:
    """Check the reviews file, the findings file and the summary's tables."""
    reviews = {r.instance_id: r for r in (saved(2), saved(1))}
    path = write_review_files(tmp_path, SETTINGS, PRS, reviews, FINDINGS)
    assert path == tmp_path / "raw" / "reviews.jsonl"
    predictions = load_predictions(path)
    assert [(p.instance_id, p.review) for p in predictions] == [
        (PRS[0].instance_id, "1. review of PR 1"),
        (PRS[1].instance_id, "1. review of PR 2"),
    ]
    lines = (tmp_path / "raw" / "findings.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    summary = (tmp_path / REVIEW_SUMMARY).read_text(encoding="utf-8")
    assert "2 findings, 1.00 per reviewed PR" in summary
    assert "Functional (F): 1; evolvability (E): 1" in summary
    assert "| F.4 | Check | 1 |" in summary
    assert f"| {PRS[1].instance_id} | 1 |" in summary
    assert "1 PRs trimmed" in summary
