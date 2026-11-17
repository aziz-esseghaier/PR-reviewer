"""Writing an evidence run's results folder: settings, a summary, the proving tests.

results/<name>/ gets:
- evidence_config.json: the run's settings. In git.
- evidence_summary.md: tiers by kind (functional or evolvability), the proof rate, how the test
  writer and the verifier ended, the cost, and each tier set against the judge's verdict on the
  same finding in the source run (a hit on a known issue, or a false alarm). In git.
- proofs/<PR>__<finding>.py: each proving test, with the claim on top. In git.
- raw/evidence.jsonl: every candidate's record. Not in git.

The judge's verdicts come from the source run's scores (pr_scores), so no new judge call is
made: the judge numbered the review's points in order, and point k is the run's finding fk.

Imported by other modules; the file isn't run directly.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import TupleRow

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.core.config import project_root
from pr_reviewer.evidence.inputs import Candidate

# The summary and settings files.
EVIDENCE_SUMMARY = "evidence_summary.md"
EVIDENCE_CONFIG = "evidence_config.json"

# How the judge saw a candidate in the source run.
LABELS = ("hit", "false_alarm", "clean", "unknown")
LABEL_NAMES = {
    "hit": "Hit (matches a known issue)",
    "false_alarm": "False alarm on a PR with issues",
    "clean": "False alarm on a clean PR",
    "unknown": "Not scored",
}

# How the test writer ended, in table order.
TEST_RESULTS = ("proof", "no_proof", "broken", "budget", "invalid", "no_reply", "not_tried")


def _hit_ids(gt_points: Sequence[Any]) -> set[str]:
    """Collect the predicted points the judge matched to a known issue.

    Args:
        gt_points (Sequence[Any]): The judgment's known issues, each with "hit" and "hit_by".

    Returns:
        set[str]: Point IDs such as "PRED-POINT-2".
    """
    hits: set[str] = set()
    for point in gt_points:
        if not isinstance(point, dict) or str(point.get("hit", "")).upper() != "YES":
            continue
        by = point.get("hit_by")
        items = by if isinstance(by, list) else str(by or "").split(",")
        hits.update(str(item).strip() for item in items if str(item).strip())
    return hits


def judge_labels(
    conn: psycopg.Connection[TupleRow],
    source_run: str,
    candidates: Mapping[str, Sequence[Candidate]],
    prs: Mapping[str, Instance],
) -> dict[tuple[str, str], str]:
    """Label each candidate with the judge's verdict on it in the source run.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        source_run (str): The scored review run.
        candidates (Mapping[str, Sequence[Candidate]]): Its candidates by PR, in order.
        prs (Mapping[str, Instance]): The PRs, to tell clean ones.

    Returns:
        dict[tuple[str, str], str]: By (PR, finding ID): "hit", "false_alarm", "clean" or
        "unknown" (the PR wasn't scored, or the judge's points don't line up with the
        findings).
    """
    rows = conn.execute(
        "SELECT instance_id, judgment FROM pr_scores WHERE experiment = %s", (source_run,)
    ).fetchall()
    judgments = {pr: judgment for pr, judgment in rows}
    labels: dict[tuple[str, str], str] = {}
    for pr, found in candidates.items():
        judgment = judgments.get(pr) or {}
        preds = judgment.get("pred_points") or []
        clean = pr in prs and not prs[pr].changes
        hits = _hit_ids(judgment.get("gt_points") or [])
        lined_up = bool(judgment) and len(preds) == len(found)
        for number, candidate in enumerate(found):
            key = (pr, candidate.finding_id)
            if not lined_up:
                labels[key] = "unknown"
            elif clean:
                labels[key] = "clean"
            else:
                point = preds[number].get("id") if isinstance(preds[number], dict) else None
                labels[key] = "hit" if point in hits else "false_alarm"
    return labels


def _share(part: int, whole: int) -> str:
    """Write a count and its share of a whole.

    Args:
        part (int): The count.
        whole (int): The whole.

    Returns:
        str: Such as "12 of 111 (10.8%)".
    """
    percent = f" ({100 * part / whole:.1f}%)" if whole else ""
    return f"{part} of {whole}{percent}"


def _tier_name(record: Mapping[str, Any]) -> str:
    """Name a record's tier, splitting tier 1 by proof kind.

    Args:
        record (Mapping[str, Any]): The record.

    Returns:
        str: Such as "1 (executed)", "1 (new code)" or "3".
    """
    if record["tier"] == 1:
        return "1 (new code)" if record.get("proof") == "new_code" else "1 (executed)"
    return str(record["tier"])


TIER_ROWS = ("1 (executed)", "1 (new code)", "2", "3", "4")


def summary_markdown(
    settings: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Candidate]],
    labels: Mapping[tuple[str, str], str],
) -> str:
    """Write the run's summary.

    Args:
        settings (Mapping[str, Any]): The run's settings.
        records (Sequence[Mapping[str, Any]]): Every candidate's record so far.
        candidates (Mapping[str, Sequence[Candidate]]): The candidates in scope, by PR.
        labels (Mapping[tuple[str, str], str]): The judge's label of each candidate.

    Returns:
        str: The Markdown text.
    """
    scope = {(pr, c.finding_id): c for pr, found in candidates.items() for c in found}
    in_scope = [r for r in records if (r["instance_id"], r["finding_id"]) in scope]
    done = [r for r in in_scope if r["status"] == "done"]
    failed = len(in_scope) - len(done)
    functional = [r for r in done if r["functional"]]
    testable = [r for r in done if scope[(r["instance_id"], r["finding_id"])].testable]
    lines = [
        f"# Evidence run {settings['name']}",
        "",
        f"Candidates from `{settings['source_run']}`, checked with {settings['profile']} "
        f"({settings['model']}) on {settings.get('date', '')}.",
        "",
        "## Progress",
        "",
        f"- Candidates checked: {_share(len(done), len(scope))}; failed (to try again): {failed}",
        f"- PRs: {len({pr for pr, _ in scope})}",
        "",
        "## Tiers",
        "",
        "| Tier | Functional | Evolvability | All |",
        "|---|---|---|---|",
    ]
    for tier in TIER_ROWS:
        f = sum(_tier_name(r) == tier for r in functional)
        a = sum(_tier_name(r) == tier for r in done)
        lines.append(f"| {tier} | {f} | {a - f} | {a} |")
    proven = [r for r in done if r["tier"] == 1]
    lines += [
        "",
        "## Proof rate",
        "",
        f"- Functional candidates proven by a test (tier 1): "
        f"{_share(sum(r['tier'] == 1 for r in functional), len(functional))}",
        f"- Testable ones (functional, naming an exception): "
        f"{_share(sum(r['tier'] == 1 for r in testable), len(testable))}",
        f"- New-code proofs among them: {sum(r.get('proof') == 'new_code' for r in proven)}",
        "",
        "## Test writer",
        "",
        "| How it ended | Candidates |",
        "|---|---|",
    ]
    results = Counter(r["test_result"] for r in done)
    for result in TEST_RESULTS:
        lines.append(f"| {result} | {results[result]} |")
    attempts = [a for r in done for a in (r["details"] or {}).get("attempts", [])]
    reasons = Counter(a.get("reason", "") for a in attempts if a.get("kind") == "no_proof")
    lines += [
        "",
        f"- Tests written: {len(attempts)}; refused by our checks before running: "
        f"{sum(a.get('kind') == 'invalid' for a in attempts)}; broken when run: "
        f"{sum(a.get('kind') == 'broken' for a in attempts)}",
    ]
    if reasons:
        lines.append(
            "- Why runs proved nothing: "
            + "; ".join(f"{reason} ({n})" for reason, n in reasons.most_common(8))
        )
    verdicts = Counter(r.get("verdict") or "not asked" for r in done)
    lines += [
        "",
        "## Verifier",
        "",
        "| Verdict | " + " | ".join(LABEL_NAMES[label] for label in LABELS) + " |",
        "|---|" + "---|" * len(LABELS),
    ]
    for verdict in ("upheld", "refuted", "deliberate", "not asked"):
        if not verdicts[verdict]:
            continue
        row = [
            sum(
                (r.get("verdict") or "not asked") == verdict
                and labels.get((r["instance_id"], r["finding_id"])) == label
                for r in done
            )
            for label in LABELS
        ]
        lines.append(f"| {verdict} | " + " | ".join(str(n) for n in row) + " |")
    confirmed = sum(bool(r.get("tool_confirmed")) for r in done)
    lines += [
        "",
        f"- Static warnings confirmed as the same problem: {confirmed} candidates",
        "",
        "## Tier against the judge",
        "",
        "Each candidate's tier, against how the judge saw the same finding in the source run.",
        "",
        "| Tier | " + " | ".join(LABEL_NAMES[label] for label in LABELS) + " |",
        "|---|" + "---|" * len(LABELS),
    ]
    for tier in TIER_ROWS:
        row = [
            sum(
                _tier_name(r) == tier and labels.get((r["instance_id"], r["finding_id"])) == label
                for r in done
            )
            for label in LABELS
        ]
        lines.append(f"| {tier} | " + " | ".join(str(n) for n in row) + " |")
    calls = sum(int(r["calls"]) for r in in_scope)
    tokens_in = sum(int(r["input_tokens"]) for r in in_scope)
    tokens_out = sum(int(r["output_tokens"]) for r in in_scope)
    minutes = sum(float(r["docker_minutes"]) for r in in_scope)
    lines += [
        "",
        "## Cost",
        "",
        f"- Model replies: {calls} ({calls / max(1, len(in_scope)):.1f} per candidate)",
        f"- Tokens: {tokens_in:,} in, {tokens_out:,} out "
        f"({(tokens_in + tokens_out) / max(1, len(in_scope)):,.0f} per candidate)",
        f"- Test runs in Docker: {minutes:.0f} minutes (stored runs count 0)",
        "",
        "## Proven findings",
        "",
    ]
    if not proven:
        lines.append("None yet.")
    else:
        lines += ["| PR | Finding | Kind | Claim | Judge | Test |", "|---|---|---|---|---|---|"]
        for r in proven:
            claim = scope[(r["instance_id"], r["finding_id"])].claim
            summary = claim.summary.replace("|", "\\|")
            label = labels.get((r["instance_id"], r["finding_id"]), "unknown")
            test = f"proofs/{proof_name(r)}"
            lines.append(
                f"| {r['instance_id']} | {r['finding_id']} | {r.get('proof')} | {summary} | "
                f"{label} | {test} |"
            )
    return "\n".join(lines) + "\n"


def proof_name(record: Mapping[str, Any]) -> str:
    """Name a proving test's copy in the results folder.

    Args:
        record (Mapping[str, Any]): The record.

    Returns:
        str: Such as "psf__requests-2317__f3.py".
    """
    return f"{record['instance_id']}__{record['finding_id']}.py"


def write_evidence_files(
    folder: Path,
    settings: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Candidate]],
    labels: Mapping[tuple[str, str], str],
) -> None:
    """Write the run's settings, summary, proving tests and raw records.

    Args:
        folder (Path): results/<name>/.
        settings (Mapping[str, Any]): The run's settings.
        records (Sequence[Mapping[str, Any]]): Every candidate's record.
        candidates (Mapping[str, Sequence[Candidate]]): The candidates in scope, by PR.
        labels (Mapping[tuple[str, str], str]): The judge's label of each candidate.
    """
    (folder / "raw").mkdir(parents=True, exist_ok=True)
    (folder / EVIDENCE_CONFIG).write_text(
        json.dumps(dict(settings), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (folder / EVIDENCE_SUMMARY).write_text(
        summary_markdown(settings, records, candidates, labels), encoding="utf-8", newline="\n"
    )
    with (folder / "raw" / "evidence.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for record in records:
            out.write(json.dumps(dict(record), default=str) + "\n")
    scope = {(pr, c.finding_id): c for pr, found in candidates.items() for c in found}
    proofs = folder / "proofs"
    for record in records:
        key = (record["instance_id"], record["finding_id"])
        if record["tier"] != 1 or not record.get("test_path") or key not in scope:
            continue
        source = project_root() / record["test_path"]
        if not source.is_file():
            continue
        claim = scope[key].claim
        header = (
            f"# Proof ({record.get('proof')}) for {record['instance_id']} "
            f"{record['finding_id']}: {claim.file}:{claim.start_line}\n"
            f"# Claim: {claim.summary}\n# Expected: {record.get('expected_failure')}\n\n"
        )
        proofs.mkdir(exist_ok=True)
        (proofs / proof_name(record)).write_text(
            header + source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
        )
