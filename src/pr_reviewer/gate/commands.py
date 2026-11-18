"""The `pr-reviewer gate ...` commands: turn checked candidates into reviews, and report.

cli.py adds them to the `pr-reviewer` command. No model calls and no Docker: the gate reads
the candidate run's reviews and the evidence run's tiers from the database.

How to run (inside the conda env, from the project folder, with the database running):
    pr-reviewer gate run --candidates <review run> --evidence <evidence run>
                         --name <gated name> --ungated-name <ungated name>
        write results/<gated name>/raw/reviews.jsonl (only what passes configs/gate.yaml) and
        results/<ungated name>/raw/reviews.jsonl (every candidate), ready for `pr-reviewer
        score`; both cover the same PRs
    pr-reviewer gate report --candidates ... --evidence ... --name ... --ungated-name ...
        once both are scored: the headline table (results/<gated name>/gate_report.md) and
        the precision/recall curve (curve.md, curve.svg), estimated from the ungated scores
Both take `--split test --unlock-test-set` for the final test-set run.
"""

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pr_reviewer.benchmark.dataset import Instance, load_dataset
from pr_reviewer.benchmark.results import GATE_CONFIG_FILE, results_folder
from pr_reviewer.benchmark.scoring import Prediction, save_predictions
from pr_reviewer.benchmark.source import DATASET, default_dataset_path
from pr_reviewer.benchmark.split import (
    SPLIT_SHA256,
    add_split_arguments,
    chosen_prs,
    load_split,
    split_path,
)
from pr_reviewer.core.config import ConfigError
from pr_reviewer.core.schemas import Finding, render_review
from pr_reviewer.db.connection import connect
from pr_reviewer.evidence.inputs import load_candidates, run_settings
from pr_reviewer.evidence.store import load_records
from pr_reviewer.gate.chart import curve_svg
from pr_reviewer.gate.curve import (
    CurvePoint,
    confidence_curve,
    evidence_curve,
    policy_point,
    ungated_point,
)
from pr_reviewer.gate.policy import Checked, GatePolicy, gate, load_policy
from pr_reviewer.reviewer.compare import (
    kind_scores,
    load_judgments,
    official_categories,
    subset_metrics,
)
from pr_reviewer.reviewer.store import load_reviews
from pr_reviewer.setup_agent.store import usable_prs

# The gate's files in the gated run's folder (score writes config.json and summary.md there;
# GATE_CONFIG_FILE, from results.py, marks a folder the gate made as free to score into).
GATE_SUMMARY = "gate_summary.md"
GATE_REPORT = "gate_report.md"


def _write(path: Path, text: str) -> None:
    """Write a text file with LF line endings, making its folder.

    Args:
        path (Path): The file.
        text (str): The text.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _load(
    args: argparse.Namespace,
) -> tuple[list[Instance], dict[str, list[Checked]], dict[str, str], dict[str, Any]]:
    """Read the run's PRs, each candidate with its tier, and the ungated reviews.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `candidates`, `evidence`, `split`,
            `unlock_test_set` and `data`.

    Returns:
        tuple[list[Instance], dict[str, list[Checked]], dict[str, str], dict[str, Any]]: The
        PRs of the candidate run, in dataset order; their checked candidates by PR; the
        candidate run's review of each PR (the ungated review); and facts about the inputs,
        for the config.

    Raises:
        ConfigError: If the runs don't belong together or aren't finished.
    """
    instances = load_dataset(args.data or default_dataset_path(), sha256=DATASET.sha256)
    in_set = chosen_prs(instances, load_split(split_path(), sha256=SPLIT_SHA256), args)
    with connect() as conn:
        source = run_settings(conn, args.candidates)
        row = conn.execute(
            "SELECT source_run, config FROM evidence_runs WHERE name = %s", (args.evidence,)
        ).fetchone()
        if row is None:
            raise ConfigError(f"There's no evidence run named {args.evidence!r}.")
        evidence_source, evidence_settings = row
        candidates = load_candidates(conn, args.candidates)
        records = load_records(conn, args.evidence)
        reviews = load_reviews(conn, args.candidates)
    if source.get("split") != args.split or evidence_settings.get("split", "dev") != args.split:
        raise ConfigError(f"{args.candidates} and {args.evidence} must both be {args.split} runs.")
    if evidence_source != args.candidates:
        raise ConfigError(f"{args.evidence} checked {evidence_source}, not {args.candidates}.")
    prs = [
        pr
        for pr in in_set
        if pr.year >= source["since"] and (pr.changes or not source.get("with_issues"))
    ]
    if source.get("usable_only"):
        with connect() as conn:
            usable = usable_prs(conn)
        prs = [pr for pr in prs if pr.instance_id in usable]
    waiting = [
        pr.instance_id for pr in prs if getattr(reviews.get(pr.instance_id), "status", "") != "done"
    ]
    if waiting:
        raise ConfigError(f"{len(waiting)} PRs have no finished review in {args.candidates}.")
    tiers = {(r["instance_id"], r["finding_id"]): r for r in records}
    checked: dict[str, list[Checked]] = {}
    unchecked = 0
    failed = 0
    for pr, found in candidates.items():
        items = []
        for order, candidate in enumerate(found):
            record = tiers.get((pr, candidate.finding_id))
            if record is None:
                unchecked += 1
                continue
            # A failed check found no evidence, so it counts as tier 4.
            failed += record["status"] == "failed"
            tier = record["tier"] if record["status"] == "done" else 4
            items.append(Checked(pr, candidate.finding_id, order, candidate.claim, tier))
        checked[pr] = items
    if unchecked:
        raise ConfigError(f"{unchecked} candidates have no evidence yet; finish {args.evidence}.")
    facts = {
        "candidates_run": args.candidates,
        "evidence_run": args.evidence,
        "split": args.split,
        "since": source["since"],
        "profile": source.get("profile"),
        "usable_only": bool(source.get("usable_only")),
        "model": source.get("model"),
        "evidence_settings": {
            key: evidence_settings.get(key)
            for key in ("prompts_sha256", "matching_version", "verifier", "profile", "model")
        },
        "failed_checks": failed,
    }
    ungated = {pr.instance_id: reviews[pr.instance_id].review for pr in prs}
    return prs, checked, ungated, facts


def _summary(
    prs: Sequence[Instance],
    checked: Mapping[str, Sequence[Checked]],
    kept: Mapping[str, Sequence[Checked]],
    policy: GatePolicy,
    facts: Mapping[str, Any],
) -> str:
    """Describe what the gate kept, as Markdown.

    Args:
        prs (Sequence[Instance]): The PRs.
        checked (Mapping[str, Sequence[Checked]]): Every checked candidate, by PR.
        kept (Mapping[str, Sequence[Checked]]): The comments kept, by PR.
        policy (GatePolicy): The gate's settings.
        facts (Mapping[str, Any]): The inputs, from _load().

    Returns:
        str: The text of gate_summary.md.
    """
    clean = {pr.instance_id for pr in prs if not pr.changes}
    tiers = Counter(c.tier for found in checked.values() for c in found)
    kept_tiers = Counter(c.tier for found in kept.values() for c in found)
    eligible = sum(
        c.tier <= policy.max_tier and c.confidence >= policy.min_confidence
        for found in checked.values()
        for c in found
    )
    total_kept = sum(len(found) for found in kept.values())
    lines = [
        f"# Gate: {facts['candidates_run']} checked by {facts['evidence_run']}",
        "",
        f"- PRs: {len(prs)} ({len(prs) - len(clean)} with known issues, {len(clean)} clean)",
        f"- Policy: tier <= {policy.max_tier}, confidence >= {policy.min_confidence}, "
        f"one comment per place: {'yes' if policy.dedupe else 'no'}, "
        f"at most {policy.max_per_pr or 'any number of'} comments per PR",
        f"- Candidates: {sum(tiers.values())}; by tier: "
        + ", ".join(f"{t}: {tiers[t]}" for t in sorted(tiers)),
        f"- Passing tier and confidence: {eligible}; kept after one per place and the cap: "
        f"{total_kept}"
        + (
            " (" + ", ".join(f"tier {t}: {kept_tiers[t]}" for t in sorted(kept_tiers)) + ")"
            if kept_tiers
            else ""
        ),
        f"- PRs with a comment: {sum(bool(found) for found in kept.values())}, of which clean: "
        f"{sum(bool(found) for pr, found in kept.items() if pr in clean)}",
        f"- Candidates whose check failed (counted as tier 4): {facts['failed_checks']}",
        "",
        "## Comments kept",
        "",
    ]
    for pr in prs:
        for item in kept.get(pr.instance_id, []):
            claim = item.claim
            lines.append(
                f"- {pr.instance_id} {item.finding_id} (tier {item.tier}, confidence "
                f"{item.confidence:.2f}): {claim.file}:{claim.start_line}: {claim.summary}"
            )
    if not total_kept:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def _run(args: argparse.Namespace) -> int:
    """Write the gated and ungated reviews of the same PRs, ready to score.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `candidates`, `evidence`, `name`,
            `ungated_name`, `config`, `split`, `unlock_test_set` and `data`.

    Returns:
        int: Exit code, 0 on success.
    """
    policy = load_policy(args.config)
    prs, checked, ungated, facts = _load(args)
    kept = {pr.instance_id: gate(checked.get(pr.instance_id, []), policy) for pr in prs}
    gated = [
        Prediction(
            instance_id=pr.instance_id,
            review=render_review(
                [Finding(id=c.finding_id, claim=c.claim) for c in kept[pr.instance_id]]
            ),
        )
        for pr in prs
    ]
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    folder = results_folder(args.name)
    config = {"name": args.name, "date": date, "policy": policy.model_dump(), **facts}
    _write(folder / GATE_CONFIG_FILE, json.dumps(config, indent=2) + "\n")
    _write(folder / GATE_SUMMARY, _summary(prs, checked, kept, policy, facts))
    save_predictions(folder / "raw" / "reviews.jsonl", gated)
    rows = [{**asdict_checked(item), "kept": True} for pr in prs for item in kept[pr.instance_id]]
    _write(folder / "raw" / "kept.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
    other = results_folder(args.ungated_name)
    ungated_config = {"name": args.ungated_name, "date": date, "ungated": True, **facts}
    _write(other / GATE_CONFIG_FILE, json.dumps(ungated_config, indent=2) + "\n")
    save_predictions(
        other / "raw" / "reviews.jsonl",
        [Prediction(instance_id=pr.instance_id, review=ungated[pr.instance_id]) for pr in prs],
    )
    unlock = " --split test --unlock-test-set" if args.split == "test" else ""
    since = f" --since {facts['since']}" + (" --usable-only" if facts["usable_only"] else "")
    reviewer = facts["profile"]
    print(
        f"{sum(len(found) for found in kept.values())} comments kept on {len(prs)} PRs. Saved "
        f"{folder / GATE_SUMMARY}\nScore both with:\n"
        f"  pr-reviewer score results/{args.name}/raw/reviews.jsonl --name {args.name} "
        f"--reviewer {reviewer}{since}{unlock}\n"
        f"  pr-reviewer score results/{args.ungated_name}/raw/reviews.jsonl --name "
        f"{args.ungated_name} --reviewer {reviewer}{since}{unlock}"
    )
    return 0


def asdict_checked(item: Checked) -> dict[str, Any]:
    """Turn a checked finding into a JSON-ready dict.

    Args:
        item (Checked): The finding.

    Returns:
        dict[str, Any]: Its PR, ID, order, tier and claim.
    """
    return {
        "instance_id": item.instance_id,
        "finding_id": item.finding_id,
        "order": item.order,
        "tier": item.tier,
        "claim": item.claim.model_dump(mode="json"),
    }


def _official(judgments: Mapping[str, dict[str, Any]], label: str, cats: list[str]) -> str:
    """Write one row of the headline table from official judgments.

    Args:
        judgments (Mapping[str, dict[str, Any]]): A scored run's judgments, by PR.
        label (str): The row's name.
        cats (list[str]): Every SWR-Bench category code.

    Returns:
        str: A Markdown table row.
    """
    items = list(judgments.values())
    metrics = subset_metrics(items, cats)
    precision_f, recall_f, f1_f = kind_scores(items, cats, "F")
    precision_e, recall_e, f1_e = kind_scores(items, cats, "E")
    clean = [j for j in items if not j["change_introduced"]]
    points = sum(len(j["pred_points"]) for j in items)
    clean_points = sum(len(j["pred_points"]) for j in clean)
    flagged = sum(bool(j["pred_points"]) for j in clean)
    return (
        f"| {label} | {points} | {metrics.hits} | {metrics.precision:.3f} | "
        f"{metrics.recall:.3f} | {metrics.f1:.3f} | {precision_f:.3f} | {recall_f:.3f} | "
        f"{f1_f:.3f} | {precision_e:.3f} | {recall_e:.3f} | {f1_e:.3f} | "
        f"{clean_points} | {flagged} of {len(clean)} | {metrics.accuracy:.3f} |"
    )


def _curve_rows(points: Sequence[CurvePoint]) -> list[str]:
    """Write a curve's points as Markdown table rows.

    Args:
        points (Sequence[CurvePoint]): The points.

    Returns:
        list[str]: The header and one row per point.
    """
    lines = [
        "| Cutoff | Comments | Hits | False alarms | On clean PRs | Precision | Recall | F1 | "
        "Precision F | Recall F | F1 F |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in points:
        lines.append(
            f"| {p.label} | {p.comments} | {p.hits} | {p.false_alarms} | {p.clean_false_alarms} "
            f"| {p.precision:.3f} | {p.recall:.3f} | {p.f1:.3f} | {p.precision_functional:.3f} | "
            f"{p.recall_functional:.3f} | {p.f1_functional:.3f} |"
        )
    return lines


def _proof_rate(args: argparse.Namespace) -> list[str]:
    """Count executed proofs among the checked candidates, by kind.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `evidence`.

    Returns:
        list[str]: Markdown lines.
    """
    with connect() as conn:
        records = load_records(conn, args.evidence)
    lines = []
    for functional, name in ((True, "functional"), (False, "evolvability")):
        group = [r for r in records if r["functional"] == functional]
        proofs = Counter(r["proof"] for r in group if r["tier"] == 1)
        share = f" ({100 * sum(proofs.values()) / len(group):.1f}%)" if group else ""
        kinds = ", ".join(f"{kind}: {count}" for kind, count in sorted(proofs.items()))
        lines.append(
            f"- {name.capitalize()} candidates: {len(group)}; executed proofs (tier 1): "
            f"{sum(proofs.values())}{share}" + (f" ({kinds})" if kinds else "")
        )
    return lines


def _report(args: argparse.Namespace) -> int:
    """Write the headline table and the precision/recall curve, once both runs are scored.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `candidates`, `evidence`, `name`,
            `ungated_name`, `config`, `split`, `unlock_test_set` and `data`.

    Returns:
        int: Exit code, 0 on success.
    """
    policy = load_policy(args.config)
    prs, checked, _, facts = _load(args)
    gated = load_judgments(results_folder(args.name))
    ungated = load_judgments(results_folder(args.ungated_name))
    wanted = {pr.instance_id for pr in prs}
    if set(gated) != wanted or set(ungated) != wanted:
        raise ConfigError("The gated and ungated scores must cover exactly the gate's PRs.")
    cats = official_categories()
    evidence_points = evidence_curve(ungated, checked, policy, cats)
    confidence_points = confidence_curve(ungated, checked, policy, cats)
    estimate = policy_point(ungated, checked, policy, cats)
    whole = ungated_point(ungated, checked, cats)
    official_ungated = subset_metrics(list(ungated.values()), cats)
    if (whole.hits, whole.false_alarms) != (official_ungated.hits, official_ungated.false_alarms):
        raise ConfigError("Cutting nothing from the ungated judgments changed its scores.")
    folder = results_folder(args.name)
    header = (
        "| Review | Comments | Hits | Precision | Recall | F1 | Precision F | Recall F | F1 F | "
        "Precision E | Recall E | F1 E | False alarms on clean PRs | Clean PRs flagged | "
        "PR accuracy |"
    )
    lines = [
        f"# Gated against ungated: {facts['split']} PRs from {facts['since']} on",
        "",
        f"- PRs: {len(prs)} ({sum(bool(pr.changes) for pr in prs)} with known issues, "
        f"{sum(not pr.changes for pr in prs)} clean)",
        f"- Candidates: {facts['candidates_run']}; evidence: {facts['evidence_run']}; model "
        f"{facts['model']}",
        f"- Gate: tier <= {policy.max_tier}, confidence >= {policy.min_confidence}, one comment "
        f"per place: {'yes' if policy.dedupe else 'no'}, at most "
        f"{policy.max_per_pr or 'any number of'} per PR",
        "- Scores: SWR-Bench's official scorer and judge (F = functional, E = evolvability).",
        "",
        header,
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        _official(ungated, "Ungated (every candidate)", cats),
        _official(gated, "Gated", cats),
        "",
        f"The gate estimated from the ungated scores: {estimate.comments} comments, "
        f"{estimate.hits} hits, precision {estimate.precision:.3f}, recall {estimate.recall:.3f}.",
        "",
        "## Proof rate",
        "",
        *_proof_rate(args),
        "",
        "## Precision/recall curve (estimated from the ungated scores)",
        "",
        "Sorted by evidence tier, then confidence:",
        "",
        *_curve_rows(evidence_points),
        "",
        "Sorted by the reviewer's confidence alone (no evidence):",
        "",
        *_curve_rows(confidence_points),
        "",
        "The curve: curve.svg.",
    ]
    _write(folder / GATE_REPORT, "\n".join(lines) + "\n")
    _write(
        folder / "curve.svg",
        curve_svg(
            evidence_points, confidence_points, whole, subset_metrics(list(gated.values()), cats)
        ),
    )
    raw = {
        "evidence": [asdict(p) for p in evidence_points],
        "confidence": [asdict(p) for p in confidence_points],
        "gate_estimate": asdict(estimate),
        "ungated": asdict(whole),
    }
    _write(folder / "raw" / "curve.json", json.dumps(raw, indent=2) + "\n")
    print(f"Saved {folder / GATE_REPORT} and {folder / 'curve.svg'}")
    return 0


def _add_options(parser: argparse.ArgumentParser) -> None:
    """Add the options both gate commands take.

    Args:
        parser (argparse.ArgumentParser): The command's parser.
    """
    parser.add_argument("--candidates", required=True, help="the candidate review run")
    parser.add_argument("--evidence", required=True, help="the evidence run that checked it")
    parser.add_argument("--name", required=True, help="the gated review's name and folder")
    parser.add_argument(
        "--ungated-name", required=True, help="the ungated review's name and folder"
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="the gate's settings (default: configs/gate.yaml)"
    )
    add_split_arguments(parser)


def add_commands(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
    data_option: argparse.ArgumentParser,
) -> None:
    """Add `pr-reviewer gate ...` and its subcommands to the command line.

    Args:
        commands (argparse._SubParsersAction[argparse.ArgumentParser]): The top-level
            subcommands of `pr-reviewer`.
        data_option (argparse.ArgumentParser): Parent parser with the `--data` option.
    """
    gate_cmd = commands.add_parser("gate", help="keep only well-evidenced findings, and report")
    sub = gate_cmd.add_subparsers(metavar="GATE_COMMAND", required=True)
    run = sub.add_parser(
        "run", parents=[data_option], help="write the gated and ungated reviews (no model calls)"
    )
    _add_options(run)
    run.set_defaults(handler=_run)
    report = sub.add_parser(
        "report", parents=[data_option], help="headline table and curve, once both are scored"
    )
    _add_options(report)
    report.set_defaults(handler=_report)
