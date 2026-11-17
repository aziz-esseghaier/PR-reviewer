"""Evidence runs in PostgreSQL (migration 0011): each candidate's tier, cost and evidence.

A run is the evidence agent going over a review run's candidates under one name, with fixed
settings. Each candidate's latest result is kept in candidate_evidence, so a stopped run knows
which candidates are done, and results can be counted by tier, proof or verdict with SQL.

Imported by other modules; the file isn't run directly.
"""

from typing import Any

import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from pr_reviewer.core.config import ConfigError
from pr_reviewer.evidence.graph import EvidenceRecord


def start_evidence_run(
    conn: psycopg.Connection[TupleRow],
    name: str,
    *,
    source_run: str,
    profile: str,
    model: str,
    settings: dict[str, Any],
) -> None:
    """Record an evidence run, or check that an existing one has the same settings.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        name (str): The run's name.
        source_run (str): The review run its candidates come from.
        profile (str): The model profile.
        model (str): The model ID.
        settings (dict[str, Any]): Everything that decides the results.

    Raises:
        ConfigError: If a run with this name exists with other settings.
    """
    row = conn.execute("SELECT config FROM evidence_runs WHERE name = %s", (name,)).fetchone()
    if row is not None:
        if row[0] != settings:
            changed = sorted(
                key
                for key in settings.keys() | row[0].keys()
                if settings.get(key) != row[0].get(key)
            )
            raise ConfigError(
                f"The evidence run {name!r} was made with other settings ({', '.join(changed)}). "
                "Pick another --name to run with these."
            )
        return
    conn.execute(
        """
        INSERT INTO evidence_runs (name, source_run, profile, model, config)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (name, source_run, profile, model, Jsonb(settings)),
    )


def evidence_status(
    conn: psycopg.Connection[TupleRow], run: str
) -> dict[tuple[str, str], tuple[str, int]]:
    """Give how far each candidate of a run got.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        run (str): The run's name.

    Returns:
        dict[tuple[str, str], tuple[str, int]]: By (PR, finding ID), its status ("done" or
        "failed") and how many times it failed. Candidates not checked yet are left out.
    """
    rows = conn.execute(
        "SELECT instance_id, finding_id, status, failures FROM candidate_evidence WHERE run = %s",
        (run,),
    ).fetchall()
    return {(pr, finding): (status, failures) for pr, finding, status, failures in rows}


def save_record(conn: psycopg.Connection[TupleRow], run: str, record: EvidenceRecord) -> None:
    """Save one candidate's result, replacing an earlier one in the same run.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        run (str): The run's name.
        record (EvidenceRecord): The result. Its sample is how many times it failed before.
    """
    failures = record.sample + (1 if record.status == "failed" else 0)
    conn.execute(
        """
        INSERT INTO candidate_evidence (run, instance_id, finding_id, status, failures,
                                        functional, expected_failure, tier, proof, test_result,
                                        verdict, tool_confirmed, test_path, calls, input_tokens,
                                        output_tokens, docker_minutes, evidence, details)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (run, instance_id, finding_id) DO UPDATE SET
            status = EXCLUDED.status, failures = EXCLUDED.failures,
            functional = EXCLUDED.functional, expected_failure = EXCLUDED.expected_failure,
            tier = EXCLUDED.tier, proof = EXCLUDED.proof, test_result = EXCLUDED.test_result,
            verdict = EXCLUDED.verdict, tool_confirmed = EXCLUDED.tool_confirmed,
            test_path = EXCLUDED.test_path, calls = EXCLUDED.calls,
            input_tokens = EXCLUDED.input_tokens, output_tokens = EXCLUDED.output_tokens,
            docker_minutes = EXCLUDED.docker_minutes, evidence = EXCLUDED.evidence,
            details = EXCLUDED.details, updated_at = now()
        """,
        (
            run,
            record.instance_id,
            record.finding_id,
            record.status,
            failures,
            record.functional,
            record.expected_failure,
            record.tier,
            record.proof,
            record.test_result,
            record.verdict,
            record.tool_confirmed,
            record.test_path,
            record.calls,
            record.input_tokens,
            record.output_tokens,
            record.docker_minutes,
            Jsonb(record.evidence),
            Jsonb(record.details),
        ),
    )


def load_records(conn: psycopg.Connection[TupleRow], run: str) -> list[dict[str, Any]]:
    """Read every candidate result of a run.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        run (str): The run's name.

    Returns:
        list[dict[str, Any]]: One dict per candidate with the table's columns, ordered by
        PR and finding.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT instance_id, finding_id, status, failures, functional, expected_failure,
                   tier, proof, test_result, verdict, tool_confirmed, test_path, calls,
                   input_tokens, output_tokens, docker_minutes, evidence, details
            FROM candidate_evidence WHERE run = %s
            ORDER BY instance_id, length(finding_id), finding_id
            """,
            (run,),
        )
        names = [column.name for column in cursor.description or []]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
