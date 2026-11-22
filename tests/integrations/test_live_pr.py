"""Tests for turning a live pull request into an Instance (integrations/live_pr.py).

How to run (from the project folder):
    pytest tests/integrations/test_live_pr.py
"""

import datetime as dt

import pytest

from pr_reviewer.integrations.live_pr import (
    LivePrError,
    build_instance,
    is_fork,
    live_instance_id,
)
from pr_reviewer.sandbox.repos import pr_number
from tests.integrations.fakes import BASE, FIRST, HEAD, REPO, commit, pull, two_commits


def test_a_pull_request_becomes_a_clean_instance_reviewed_at_its_last_commit() -> None:
    """Check the ID, the commits, the base, the review time and the empty known issues."""
    commits, details = two_commits()
    pr = build_instance(REPO, pull(), commits, details)
    assert pr.instance_id == "octo__demo-7"
    assert pr_number(pr) == 7
    assert pr.head_commit == HEAD
    assert pr.base_commit == BASE
    assert [c.sha for c in pr.pr_commits] == [FIRST, HEAD]
    assert pr.created_at == dt.datetime(2026, 11, 21, 9, 30, tzinfo=dt.UTC)
    assert pr.changes == () and pr.group == "clean"
    # The PR's text is kept as data for the reviewer, whatever it says.
    assert "Ignore previous instructions" in pr.pr_statement


def test_files_without_a_patch_are_left_out() -> None:
    """Check a binary file (no patch) isn't part of the reviewed diff."""
    commits, details = two_commits()
    pr = build_instance(REPO, pull(), commits, details)
    assert [p.file for p in pr.pr_commits[1].diff] == ["README.md"]


def test_a_merge_commit_is_refused() -> None:
    """Check a PR that merged its target branch in can't be reviewed."""
    commits, details = two_commits()
    details[HEAD] = {**details[HEAD], "parents": [{"sha": FIRST}, {"sha": "d" * 40}]}
    with pytest.raises(LivePrError, match="merge commit"):
        build_instance(REPO, pull(), commits, details)


def test_commits_out_of_date_order_are_refused() -> None:
    """Check commits whose dates go backwards can't be reviewed."""
    commits, details = two_commits()
    details[HEAD] = commit(HEAD, FIRST, "2026-11-19T00:00:00Z", [])
    with pytest.raises(LivePrError, match="can't be reviewed"):
        build_instance(REPO, pull(), commits, details)


def test_a_pull_request_without_commits_is_refused() -> None:
    """Check an empty commit list is refused."""
    with pytest.raises(LivePrError, match="no commits"):
        build_instance(REPO, pull(), [], {})


def test_forks_are_told_apart() -> None:
    """Check a branch in another repo, or in a deleted one, counts as a fork."""
    assert not is_fork(pull())
    assert is_fork(pull(fork=True))
    deleted = pull()
    deleted["head"]["repo"] = None
    assert is_fork(deleted)


def test_a_repo_name_that_could_change_the_path_is_refused() -> None:
    """Check only plain owner/name repos are accepted."""
    with pytest.raises(Exception, match="Not a GitHub repo name"):
        live_instance_id("octo/../admin", 1)
