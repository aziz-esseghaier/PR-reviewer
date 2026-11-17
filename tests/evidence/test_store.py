"""Tests for keeping evidence runs in PostgreSQL (evidence/store.py) and the judge's labels.

Checks that a run's settings are fixed once recorded, that a candidate's record is saved, read
back and replaced, that failures are counted, and that each candidate gets the judge's
verdict from the source run's scores (evidence/report.py). Marked `integration`: they use a
throwaway database and are skipped when Postgres isn't running.

How to run (from the project folder):
    pytest tests/evidence/test_store.py
"""

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from pr_reviewer.core.config import ConfigError
from pr_reviewer.core.schemas import Claim, Finding
from pr_reviewer.db.migrate import migrate
from pr_reviewer.evidence.graph import EvidenceRecord
from pr_reviewer.evidence.inputs import Candidate, load_candidates, run_settings
from pr_reviewer.evidence.report import judge_labels, summary_markdown
from pr_reviewer.evidence.store import (
    evidence_status,
    load_records,
    save_record,
    start_evidence_run,
)
from pr_reviewer.reviewer.naive import PrReview
from pr_reviewer.reviewer.store import save_review, start_run
from tests.reviewer.helpers import finding, make_pr

pytestmark = pytest.mark.integration

ISSUES = "psf__requests-1"
CLEAN = "psf__requests-2"
SETTINGS = {"model": "m", "budget": {"writer_replies": 3}}


@pytest.fixture
def migrated(test_db_url: str) -> Iterator[psycopg.Connection[TupleRow]]:
    """Open a migrated test database with two PRs, a scored review run and an evidence run.

    Args:
        test_db_url (str): Connection string of the empty throwaway database.

    Yields:
        psycopg.Connection[TupleRow]: Autocommit connection.
    """
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        migrate(conn)
        conn.execute(
            "INSERT INTO bench_prs VALUES (%s, 'psf/requests', 2022, 'functional', 'dev', 1, 0),"
            " (%s, 'psf/requests', 2022, 'clean', 'dev', 0, 0)",
            (ISSUES, CLEAN),
        )
        start_run(
            conn,
            "cands",
            reviewer="candidates",
            profile="p",
            model="m",
            split="dev",
            settings={"split": "dev"},
        )
        for pr, lines in ((ISSUES, (1, 2)), (CLEAN, (3,))):
            claims = [
                Claim.model_validate(finding(line=line, expected_failure="None"))
                if line == 2
                else Claim.model_validate(finding(line=line))
                for line in lines
            ]
            save_review(
                conn,
                "cands",
                PrReview(
                    instance_id=pr,
                    ok=True,
                    findings=tuple(
                        Finding(id=f"f{n}", claim=c) for n, c in enumerate(claims, start=1)
                    ),
                    review="1. x",
                    tries=1,
                    input_tokens=1,
                    output_tokens=1,
                    left_out=(),
                    cut=False,
                    full_chars=1,
                    log=(),
                ),
                0,
            )
        conn.execute(
            "INSERT INTO experiments (name, split, reviewer, judge_model, config) "
            "VALUES ('cands', 'dev', 'm', 'j', '{}')"
        )
        judgments = {
            ISSUES: {
                "gt_points": [{"id": "GT-POINT-1", "hit": "YES", "hit_by": "PRED-POINT-2"}],
                "pred_points": [{"id": "PRED-POINT-1"}, {"id": "PRED-POINT-2"}],
            },
            CLEAN: {"gt_points": None, "pred_points": [{"id": "PRED-POINT-1"}]},
        }
        for pr, judgment in judgments.items():
            conn.execute(
                "INSERT INTO pr_scores VALUES ('cands', %s, false, 1, 0, 1, 0, %s)",
                (pr, Jsonb(judgment)),
            )
        start_evidence_run(
            conn, "ev", source_run="cands", profile="p", model="m", settings=SETTINGS
        )
        yield conn


def record(finding_id: str = "f1", **changes: Any) -> EvidenceRecord:
    """Build a candidate's record.

    Args:
        finding_id (str): The candidate.
        **changes (Any): Fields to change.

    Returns:
        EvidenceRecord: The record.
    """
    fields: dict[str, Any] = {
        "instance_id": ISSUES,
        "finding_id": finding_id,
        "sample": 0,
        "status": "done",
        "functional": True,
        "expected_failure": "ZeroDivisionError",
        "tier": 1,
        "proof": "executed",
        "test_result": "proof",
        "verdict": None,
        "tool_confirmed": False,
        "test_path": "data/evidence/ev/x/f1/attempt-1.py",
        "calls": 2,
        "input_tokens": 300,
        "output_tokens": 100,
        "docker_minutes": 3.5,
        "evidence": [{"kind": "execution"}],
        "details": {"attempts": [{"n": 1, "kind": "proof"}]},
    }
    fields.update(changes)
    return EvidenceRecord.model_validate(fields)


def test_a_runs_settings_are_fixed_once_recorded(migrated: psycopg.Connection[TupleRow]) -> None:
    """Check the same settings are accepted again and different ones refused."""
    start_evidence_run(
        migrated, "ev", source_run="cands", profile="p", model="m", settings=SETTINGS
    )
    with pytest.raises(ConfigError, match="budget"):
        start_evidence_run(
            migrated, "ev", source_run="cands", profile="p", model="m", settings={"model": "m"}
        )


def test_records_are_saved_replaced_and_counted(migrated: psycopg.Connection[TupleRow]) -> None:
    """Check a failed record counts a failure, and a later one replaces it."""
    save_record(migrated, "ev", record(status="failed", tier=4, proof=None))
    assert evidence_status(migrated, "ev") == {(ISSUES, "f1"): ("failed", 1)}
    save_record(migrated, "ev", record(sample=1))
    save_record(migrated, "ev", record("f2", tier=3, proof=None, verdict="upheld"))
    assert evidence_status(migrated, "ev") == {
        (ISSUES, "f1"): ("done", 1),
        (ISSUES, "f2"): ("done", 0),
    }
    rows = load_records(migrated, "ev")
    assert [(r["finding_id"], r["tier"], r["verdict"]) for r in rows] == [
        ("f1", 1, None),
        ("f2", 3, "upheld"),
    ]
    assert rows[0]["details"]["attempts"][0]["kind"] == "proof"


def test_candidates_come_back_cleaned_up(migrated: psycopg.Connection[TupleRow]) -> None:
    """Check candidates are read by PR and in order, with a "None" exception read as none."""
    found = load_candidates(migrated, "cands")
    assert [c.finding_id for c in found[ISSUES]] == ["f1", "f2"]
    first, second = found[ISSUES]
    assert first.testable and first.expected == "ZeroDivisionError"
    assert second.expected is None and not second.testable
    assert run_settings(migrated, "cands") == {"split": "dev"}
    with pytest.raises(ConfigError):
        run_settings(migrated, "nope")


def test_each_candidate_gets_the_judges_verdict(migrated: psycopg.Connection[TupleRow]) -> None:
    """Check hits, false alarms, clean PRs, and the summary built on them."""
    found = load_candidates(migrated, "cands")
    prs = {ISSUES: make_pr(issues=("F.2 Logic",)), CLEAN: make_pr(number=2)}
    labels = judge_labels(migrated, "cands", found, prs)
    assert labels == {
        (ISSUES, "f1"): "false_alarm",
        (ISSUES, "f2"): "hit",
        (CLEAN, "f1"): "clean",
    }
    save_record(migrated, "ev", record())
    save_record(migrated, "ev", record("f2", tier=3, proof=None, verdict="upheld"))
    text = summary_markdown(
        {"name": "ev", "source_run": "cands", "profile": "p", "model": "m"},
        load_records(migrated, "ev"),
        found,
        labels,
    )
    assert "Candidates checked: 2 of 3 (66.7%)" in text
    assert "| 1 (executed) | 1 | 0 | 1 |" in text
    assert "| 3 | 1 | 0 | 0 | 0 |" in text, "tier 3: one hit"
    assert "| upheld | 1 | 0 | 0 | 0 |" in text
    assert f"| {ISSUES} | f1 | executed |" in text


def test_unscored_or_misaligned_prs_are_unknown(migrated: psycopg.Connection[TupleRow]) -> None:
    """Check a PR whose points don't line up with its findings gets no verdict."""
    extra = {
        ISSUES: [
            *load_candidates(migrated, "cands")[ISSUES],
            Candidate(ISSUES, "f9", Claim.model_validate(finding())),
        ]
    }
    labels = judge_labels(migrated, "cands", extra, {ISSUES: make_pr(issues=("F.2 Logic",))})
    assert set(labels.values()) == {"unknown"}
