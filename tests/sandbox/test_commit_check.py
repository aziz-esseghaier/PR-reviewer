"""
Tests for the commit check's results in sandbox/commit_check.py: the summary, the results
folder and the pr_checkouts table.

Uses hand-made checks, so no Docker is needed (test_checkout.py runs real checks). The test
marked `integration` uses a throwaway database and is skipped when Postgres isn't running.

How to run (from the project folder):
    pytest tests/sandbox/test_commit_check.py
"""

import json
from pathlib import Path

import psycopg
import pytest

from pr_reviewer.db.migrate import migrate
from pr_reviewer.sandbox.commit_check import (
    CommitCheck,
    save_checks,
    summary_markdown,
    write_check_results,
)

LATER = CommitCheck(
    "psf__requests-1",
    "psf/requests",
    "a" * 40,
    base_sha="b" * 40,
    base_field="c" * 40,
    base_field_relation="later",
    commits_apart=3,
    extra_files=(),
    python_files={"base": 10, "pr": 11},
)
MERGED = CommitCheck(
    "psf__requests-2",
    "psf/requests",
    "d" * 40,
    base_sha="e" * 40,
    base_field="e" * 40,
    base_field_relation="same",
    commits_apart=0,
    merge_commits=1,
    extra_files=("setup.py",),
    python_files={"base": 10, "pr": 10},
)
MISSING = CommitCheck(
    "pallets__flask-3",
    "pallets/flask",
    "f" * 40,
    base_field="0" * 40,
    missing_commits=("f" * 40,),
    problem="1 of 1 reviewed commits missing",
)
CHECKS = [LATER, MERGED, MISSING]
CONFIG = {"date": "2026-10-30", "image": "pr-reviewer-sandbox:abc"}


def test_usable_and_only_pr_changes_follow_the_findings() -> None:
    """Check a PR is usable only without a problem, and extra files mean more than its own."""
    assert LATER.usable and LATER.only_pr_changes
    assert MERGED.usable and MERGED.only_pr_changes is False
    assert not MISSING.usable and MISSING.only_pr_changes is None


def test_summary_counts_each_finding() -> None:
    """Check the summary's counts, the per-repo rows and the list of PRs to look at."""
    summary = summary_markdown(CHECKS, CONFIG)
    assert "| Usable | 2 |" in summary
    assert "| Not usable | 1 |" in summary
    assert "| the base itself | 1 |" in summary
    assert "| later on the same line (other commits came in between) | 1 |" in summary
    assert "When it's later, it's 3 commits after the base" in summary
    assert "| psf/requests | 2 | 2 | 1 | 1 | 1 |" in summary
    assert "- pallets__flask-3: not usable: 1 of 1 reviewed commits missing" in summary
    assert "- psf__requests-2: 1 merge commit(s); 1 extra file(s), such as setup.py" in summary
    assert "psf__requests-1:" not in summary


def test_results_folder_holds_config_summary_and_raw(tmp_path: Path) -> None:
    """Check the three files, with one JSON line per PR in raw/."""
    write_check_results(tmp_path, CHECKS, CONFIG)
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) == CONFIG
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").startswith("# Commit check")
    lines = (tmp_path / "raw" / "checks.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["instance_id"] for line in lines] == [c.instance_id for c in CHECKS]


@pytest.mark.integration
def test_checks_are_saved_and_replaced(test_db_url: str) -> None:
    """Check one row per PR, with the columns results are grouped by, replaced on a rerun."""
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        migrate(conn)
        for check in CHECKS:
            conn.execute(
                "INSERT INTO bench_prs VALUES (%s, %s, 2020, 'clean', 'dev', 0, 0)",
                (check.instance_id, check.repo),
            )
        save_checks(conn, CHECKS)
        save_checks(conn, [LATER])
        rows = conn.execute(
            "SELECT instance_id, usable, base_field_relation, merge_commits, only_pr_changes, "
            "problem, details->>'commits_apart' FROM pr_checkouts ORDER BY instance_id"
        ).fetchall()
    assert rows == [
        ("pallets__flask-3", False, None, 0, None, "1 of 1 reviewed commits missing", None),
        ("psf__requests-1", True, "later", 0, True, None, "3"),
        ("psf__requests-2", True, "same", 1, False, None, "0"),
    ]
