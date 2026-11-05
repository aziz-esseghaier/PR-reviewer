"""Tests for keeping review runs in PostgreSQL (reviewer/store.py).

Checks that a run's settings are fixed once recorded, that a PR's review and findings are
saved, read back and replaced when the PR is reviewed again, and that a failed review is kept
with its failure count. Marked `integration`: they use a throwaway database and are skipped
when Postgres isn't running.

How to run (from the project folder):
    pytest tests/reviewer/test_store.py
"""

from collections.abc import Iterator

import psycopg
import pytest
from psycopg.rows import TupleRow

from pr_reviewer.core.config import ConfigError
from pr_reviewer.core.schemas import Claim, Finding
from pr_reviewer.db.migrate import migrate
from pr_reviewer.reviewer.naive import PrReview
from pr_reviewer.reviewer.store import (
    load_findings,
    load_reviews,
    review_status,
    save_review,
    start_run,
)
from tests.reviewer.helpers import finding

pytestmark = pytest.mark.integration

PR_ID = "psf__requests-1"
SETTINGS = {"model": "m", "max_tokens": 3000}


@pytest.fixture
def migrated(test_db_url: str) -> Iterator[psycopg.Connection[TupleRow]]:
    """Open a fresh migrated test database with one benchmark PR and one review run.

    Args:
        test_db_url (str): Connection string of the empty throwaway database.

    Yields:
        psycopg.Connection[TupleRow]: Autocommit connection.
    """
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        migrate(conn)
        conn.execute(
            "INSERT INTO bench_prs VALUES (%s, 'psf/requests', 2022, 'clean', 'dev', 0, 0)",
            (PR_ID,),
        )
        start_run(
            conn, "run", reviewer="naive", profile="p", model="m", split="dev", settings=SETTINGS
        )
        yield conn


def review(findings: int, ok: bool = True) -> PrReview:
    """Make a finished review of the test PR.

    Args:
        findings (int): How many findings it has.
        ok (bool): Whether a reply was accepted.

    Returns:
        PrReview: The review.
    """
    claims = [Claim.model_validate(finding(line=line)) for line in range(1, findings + 1)]
    return PrReview(
        instance_id=PR_ID,
        ok=ok,
        findings=tuple(Finding(id=f"f{n}", claim=c) for n, c in enumerate(claims, start=1)),
        review="1. a point" if ok else "",
        tries=1 if ok else 3,
        input_tokens=100,
        output_tokens=50,
        left_out=("pkg/a.py (commit 1234567)",),
        cut=False,
        full_chars=9000,
        log=("review: try 1",),
    )


def test_a_runs_settings_cant_change(migrated: psycopg.Connection[TupleRow]) -> None:
    """Check the same settings are accepted again and other ones are refused, by name."""
    start_run(
        migrated, "run", reviewer="naive", profile="p", model="m", split="dev", settings=SETTINGS
    )
    with pytest.raises(ConfigError, match="max_tokens"):
        start_run(
            migrated,
            "run",
            reviewer="naive",
            profile="p",
            model="m",
            split="dev",
            settings={**SETTINGS, "max_tokens": 2000},
        )


def test_a_review_is_saved_read_back_and_replaced(migrated: psycopg.Connection[TupleRow]) -> None:
    """Check the review row, its findings, and that reviewing again replaces both."""
    save_review(migrated, "run", review(findings=2), failures=0)
    save_review(migrated, "run", review(findings=1), failures=0)
    saved = load_reviews(migrated, "run")[PR_ID]
    assert (saved.status, saved.findings, saved.trimmed, saved.left_out) == (
        "done",
        1,
        True,
        ("pkg/a.py (commit 1234567)",),
    )
    found = load_findings(migrated, "run")
    assert [(f["instance_id"], f["id"], f["claim"]["category"]) for f in found] == [
        (PR_ID, "f1", "F.4")
    ]
    row = migrated.execute("SELECT functional, tier, confidence FROM findings").fetchone()
    assert row == (True, 4, pytest.approx(0.8))


def test_a_failed_review_keeps_its_count(migrated: psycopg.Connection[TupleRow]) -> None:
    """Check a failed review is saved with no findings and its failure count."""
    save_review(migrated, "run", review(findings=0, ok=False), failures=1)
    assert review_status(migrated, "run") == {PR_ID: ("failed", 1)}
    assert load_findings(migrated, "run") == []
