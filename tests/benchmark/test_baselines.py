"""
Tests for the reviews written without a model in benchmark/baselines.py.

Checks that the no-comments reviews approve every PR in SWR-Bench's own words, that the
known-issue reviews list each PR's known issues in order, and that clean PRs get the same
review in both (so the judge's cached replies answer them the second time).

How to run (from the project folder):
    pytest tests/benchmark/test_baselines.py
"""

import json
from collections.abc import Sequence

from pr_reviewer.benchmark.baselines import (
    known_issue_review,
    known_issue_reviews,
    no_comment_reviews,
)
from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.benchmark.source import official_dir
from pr_reviewer.core.schemas import NO_ISSUES
from tests.benchmark.test_dataset import make_row


def make_pr(number: int, summaries: Sequence[str]) -> Instance:
    """Make a PR whose known issues have the given summaries.

    Args:
        number (int): PR number, used in the instance ID.
        summaries (Sequence[str]): SWR-Bench's summary of each known issue. Empty for a
            clean PR.

    Returns:
        Instance: The PR.
    """
    row = make_row(number, issues=["F.2 Logic"] * len(summaries))
    for change, summary in zip(row["changes"], summaries, strict=True):
        change["change_discussion"]["discussion_summary"] = summary
    return Instance.model_validate_json(json.dumps(row))


WITH_ISSUES = make_pr(1, ["Divides by zero for an empty cart.  ", "Leaks the file handle."])
CLEAN = make_pr(2, [])


def test_approval_is_in_swr_bench_own_words() -> None:
    """Check that the approval is the sentence SWR-Bench's review prompt asks for."""
    prompt = (official_dir() / "generation.py").read_text(encoding="utf-8")
    assert f"state that {NO_ISSUES[0].lower()}{NO_ISSUES[1:]}" in prompt


def test_no_comment_reviews_approve_every_pr() -> None:
    """Check that every PR gets the approval, in the order given."""
    reviews = no_comment_reviews([WITH_ISSUES, CLEAN])
    assert [(r.instance_id, r.review) for r in reviews] == [
        (WITH_ISSUES.instance_id, NO_ISSUES),
        (CLEAN.instance_id, NO_ISSUES),
    ]


def test_known_issue_review_numbers_each_summary() -> None:
    """Check that the review lists the known issues in order, one numbered line each."""
    assert known_issue_review(WITH_ISSUES) == (
        "1. Divides by zero for an empty cart.\n2. Leaks the file handle."
    )


def test_clean_prs_get_the_same_review_in_both() -> None:
    """Check that only PRs with issues differ, so only they need new judge calls."""
    no_comments = no_comment_reviews([WITH_ISSUES, CLEAN])
    known_issues = known_issue_reviews([WITH_ISSUES, CLEAN])
    assert known_issues[1] == no_comments[1]
    assert known_issues[0] != no_comments[0]
