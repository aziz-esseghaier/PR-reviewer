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
    assert "## Context" not in summary


def test_a_context_run_summarizes_its_context(tmp_path: Path) -> None:
    """Check the Context section: PRs with context, no room, characters and pieces by kind."""
    held = {
        "room": 8000,
        "chars": 3000,
        "included": ["changed a.py:1-9 f", "caller b.py:3-8 g"],
        "left_out": ["test t.py:1-90 test_f"],
    }
    full = {"room": 0, "chars": 0, "included": [], "left_out": ["caller b.py:3-8 g"]}
    one, two = saved(1), saved(2)
    reviews = {
        one.instance_id: SavedReview(**{**vars(one), "context": held}),
        two.instance_id: SavedReview(**{**vars(two), "context": full}),
    }
    write_review_files(tmp_path, {**SETTINGS, "reviewer": "context"}, PRS, reviews, FINDINGS)
    summary = (tmp_path / REVIEW_SUMMARY).read_text(encoding="utf-8")
    assert "PRs whose request held context: 1 of 2; 1 had no room left" in summary
    assert "Characters of context: median 3,000, largest 3,000, total 3,000" in summary
    assert "| caller | 1 | 1 |" in summary
    assert "| test | 0 | 1 |" in summary


def test_a_candidate_run_summarizes_warnings_and_lookups(tmp_path: Path) -> None:
    """Check the Candidate stage section, and a lookup round not counted as a retry."""
    asked = {
        "variant": "v1",
        "static_status": "ok",
        "warnings": 3,
        "warnings_shown": 2,
        "lookups_allowed": True,
        "lookups": ["read_file pkg/a.py 1-9", "search_symbol f"],
        "lookups_found": 1,
        "discarded": 2,
    }
    plain = {**asked, "static_status": "failed", "warnings": 0, "warnings_shown": 0}
    plain.update(lookups=[], lookups_found=0, discarded=0, lookups_allowed=False)
    one, two = saved(1), saved(2)
    reviews = {
        one.instance_id: SavedReview(**{**vars(one), "tries": 2, "details": {"candidates": asked}}),
        two.instance_id: SavedReview(**{**vars(two), "details": {"candidates": plain}}),
    }
    settings = {**SETTINGS, "reviewer": "candidates"}
    write_review_files(tmp_path, settings, PRS, reviews, FINDINGS)
    summary = (tmp_path / REVIEW_SUMMARY).read_text(encoding="utf-8")
    assert "characters of request text (PR, warnings, lookups, context)" in summary
    assert "Prompt variant: v1; candidates per PR: median 1, most 1" in summary
    assert "ran on 1 PRs, no Python line changed on 0, failed on 1; 3 warnings" in summary
    assert "allowed on 1 PRs (the others had no room left), asked on 1; 2 lookups" in summary
    assert "(1 read_file, 1 search_symbol), 1 found something; 2 candidates" in summary
    assert "Model calls: 3 (0 PRs needed more than one try)" in summary
