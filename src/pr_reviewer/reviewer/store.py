"""Review runs in PostgreSQL (migration 0008): each PR's review, and every finding.

A run is a reviewer going over a set of PRs under one name, with fixed settings. Each PR's
latest review is kept in pr_reviews, and its findings in findings, so a stopped run knows which
PRs are done, and later results can be counted by category, file or confidence with SQL.

Imported by other modules; the file isn't run directly.
"""

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from pr_reviewer.core.config import ConfigError
from pr_reviewer.core.schemas import Finding
from pr_reviewer.reviewer.naive import PrReview


def start_run(
    conn: psycopg.Connection[TupleRow],
    name: str,
    *,
    reviewer: str,
    profile: str,
    model: str,
    split: str,
    settings: dict[str, Any],
) -> None:
    """Record a review run, or check that an existing one has the same settings.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        name (str): The run's name.
        reviewer (str): Which reviewer, such as "naive".
        profile (str): The model profile.
        model (str): The model ID.
        split (str): "dev" or "test".
        settings (dict[str, Any]): Everything that decides the reviews: model settings,
            prompt fingerprint, size budget, limits.

    Raises:
        ConfigError: If a run with this name exists with other settings, since its reviews
            would mix two setups.
    """
    row = conn.execute("SELECT config FROM review_runs WHERE name = %s", (name,)).fetchone()
    if row is not None:
        if row[0] != settings:
            changed = sorted(
                key
                for key in settings.keys() | row[0].keys()
                if settings.get(key) != row[0].get(key)
            )
            raise ConfigError(
                f"The review run {name!r} was made with other settings ({', '.join(changed)}). "
                "Pick another --name to review with these."
            )
        return
    conn.execute(
        """
        INSERT INTO review_runs (name, reviewer, profile, model, split, config)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (name, reviewer, profile, model, split, Jsonb(settings)),
    )


def review_status(conn: psycopg.Connection[TupleRow], run: str) -> dict[str, tuple[str, int]]:
    """Give how far each PR of a run got.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        run (str): The run's name.

    Returns:
        dict[str, tuple[str, int]]: By PR, its status ("done" or "failed") and how many of
        its reviews failed. PRs not reviewed yet are left out.
    """
    rows = conn.execute(
        "SELECT instance_id, status, failures FROM pr_reviews WHERE run = %s", (run,)
    ).fetchall()
    return {instance_id: (status, failures) for instance_id, status, failures in rows}


def _finding_row(run: str, instance_id: str, finding: Finding) -> tuple[Any, ...]:
    """Lay out one finding as a row of the findings table.

    Args:
        run (str): The run's name.
        instance_id (str): The PR.
        finding (Finding): The finding.

    Returns:
        tuple[Any, ...]: The row's values, in the table's column order.
    """
    claim = finding.claim
    return (
        run,
        instance_id,
        finding.id,
        claim.file,
        claim.start_line,
        claim.end_line,
        claim.category,
        claim.is_functional,
        claim.confidence,
        claim.expected_failure,
        int(finding.tier),
        Jsonb(claim.model_dump(mode="json")),
        Jsonb([item.model_dump(mode="json") for item in finding.evidence]),
    )


def save_review(
    conn: psycopg.Connection[TupleRow], run: str, result: PrReview, failures: int
) -> None:
    """Save one PR's review and findings, replacing an earlier review of it in the same run.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        run (str): The run's name.
        result (PrReview): How the review ended.
        failures (int): How many reviews of this PR have failed, this one included.
    """
    details = {"left_out": list(result.left_out), "cut": result.cut, "log": list(result.log)}
    with conn.transaction(), conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM pr_reviews WHERE run = %s AND instance_id = %s",
            (run, result.instance_id),
        )
        cursor.execute(
            """
            INSERT INTO pr_reviews (run, instance_id, status, failures, tries, findings, trimmed,
                                    input_tokens, output_tokens, review, details)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run,
                result.instance_id,
                "done" if result.ok else "failed",
                failures,
                result.tries,
                len(result.findings),
                result.trimmed,
                result.input_tokens,
                result.output_tokens,
                result.review,
                Jsonb(details),
            ),
        )
        cursor.executemany(
            """
            INSERT INTO findings (run, instance_id, finding_id, file, start_line, end_line,
                                  category, functional, confidence, expected_failure, tier,
                                  claim, evidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [_finding_row(run, result.instance_id, finding) for finding in result.findings],
        )


@dataclass(frozen=True)
class SavedReview:
    """One PR's saved review.

    Attributes:
        instance_id (str): The PR.
        status (str): "done" or "failed".
        failures (int): Reviews of it that failed.
        tries (int): Replies asked for in the last review.
        findings (int): Findings in the review.
        trimmed (bool): Whether the model saw less than the whole PR.
        input_tokens (int): Prompt tokens.
        output_tokens (int): Reply tokens.
        review (str): The review text.
        left_out (tuple[str, ...]): File diffs left out to fit.
    """

    instance_id: str
    status: str
    failures: int
    tries: int
    findings: int
    trimmed: bool
    input_tokens: int
    output_tokens: int
    review: str
    left_out: tuple[str, ...]


def load_reviews(conn: psycopg.Connection[TupleRow], run: str) -> dict[str, SavedReview]:
    """Read every saved review of a run.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        run (str): The run's name.

    Returns:
        dict[str, SavedReview]: The reviews, by PR.
    """
    rows = conn.execute(
        """
        SELECT instance_id, status, failures, tries, findings, trimmed, input_tokens,
               output_tokens, review, details
        FROM pr_reviews WHERE run = %s
        """,
        (run,),
    ).fetchall()
    reviews: dict[str, SavedReview] = {}
    for (
        instance_id,
        status,
        failures,
        tries,
        findings,
        trimmed,
        input_tokens,
        output_tokens,
        review,
        details,
    ) in rows:
        reviews[instance_id] = SavedReview(
            instance_id=instance_id,
            status=status,
            failures=failures,
            tries=tries,
            findings=findings,
            trimmed=trimmed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            review=review,
            left_out=tuple(details.get("left_out", [])),
        )
    return reviews


def load_findings(conn: psycopg.Connection[TupleRow], run: str) -> list[dict[str, Any]]:
    """Read every finding of a run.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        run (str): The run's name.

    Returns:
        list[dict[str, Any]]: One object per finding: the PR, the finding's ID, its claim and
        its evidence, ordered by PR and finding.
    """
    rows = conn.execute(
        """
        SELECT instance_id, finding_id, claim, evidence FROM findings WHERE run = %s
        ORDER BY instance_id, length(finding_id), finding_id
        """,
        (run,),
    ).fetchall()
    return [
        {"instance_id": pr, "id": finding_id, "claim": claim, "evidence": evidence}
        for pr, finding_id, claim, evidence in rows
    ]
