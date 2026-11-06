"""Comparing two scored review runs on the same PRs: overall, on subsets, and PR by PR.

Both runs must be scored (`pr-reviewer score`), so each results folder holds the judge's
judgment of every PR (raw/swrbench_output.json); no judge is called again. Metrics on a
subset of PRs use the formulas of SWR-Bench's analyze_result(): PR-level accuracy, point-level
precision, recall and F1, and F1 over functional (F) and evolvability (E) points. They're
computed here because analyze_result() itself divides by zero on a subset with no PR with
issues; on all PRs, they're checked against analyze_result()'s own numbers.

Imported by other modules; the file isn't run directly.
"""

import copy
import json
import warnings
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pr_reviewer.benchmark.official import load_evaluation
from pr_reviewer.benchmark.scoring import PRScore, pr_score
from pr_reviewer.benchmark.source import DatasetError

# The judgments of a scored run, inside its results folder.
JUDGMENTS = Path("raw") / "swrbench_output.json"


@dataclass(frozen=True)
class SubsetMetrics:
    """SWR-Bench's main metrics over some PRs.

    Attributes:
        prs (int): How many PRs.
        accuracy (float): PR-level accuracy: PRs with issues flagged, clean PRs passed.
        precision (float): Point-level precision.
        recall (float): Point-level recall.
        f1 (float): Point-level F1.
        f1_functional (float): F1 over functional (F) points and issues.
        f1_evolvability (float): F1 over evolvability (E) points and issues.
        hits (int): Points that matched a known issue.
        false_alarms (int): Points that matched none.
    """

    prs: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    f1_functional: float
    f1_evolvability: float
    hits: int
    false_alarms: int


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall and F1 as SWR-Bench does (0 when undefined, 4 decimals).

    Args:
        tp (int): Hits.
        fp (int): False alarms.
        fn (int): Misses.

    Returns:
        tuple[float, float, float]: Precision, recall and F1.
    """
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _type_counts(judgment: dict[str, Any], types: Collection[str]) -> tuple[int, int, int]:
    """Count one PR's hits, false alarms and misses among some categories.

    Mirrors analyze_result()'s analyze_changetype_results(): a point counts by the category
    the judge gave it, an issue by its own.

    Args:
        judgment (dict[str, Any]): The PR's judgment.
        types (Collection[str]): The categories, such as every F one.

    Returns:
        tuple[int, int, int]: Hits, false alarms and misses.
    """
    tp = fp = fn = 0
    points = judgment["pred_points"]
    if not judgment["change_introduced"]:
        return 0, sum(p["change_category"] in types for p in points), 0
    hit_by = {issue["hit_by"] for issue in judgment["gt_points"]}
    for issue in judgment["gt_points"]:
        if issue["change_category"] in types and issue["hit"].lower() == "no":
            fn += 1
    for point in points:
        if point["change_category"] in types:
            if point["id"] in hit_by:
                tp += 1
            else:
                fp += 1
    return tp, fp, fn


def subset_metrics(
    judgments: Sequence[dict[str, Any]], categories: Collection[str]
) -> SubsetMetrics:
    """Compute SWR-Bench's main metrics over some PRs' judgments.

    Args:
        judgments (Sequence[dict[str, Any]]): The PRs' judgments.
        categories (Collection[str]): Every SWR-Bench category code, such as "F.2".

    Returns:
        SubsetMetrics: The metrics.
    """
    scores = [pr_score(judgment) for judgment in judgments]
    right = sum(s.has_issues != s.judged_good for s in scores)
    hits = sum(s.hits for s in scores)
    false_alarms = sum(s.false_alarms for s in scores)
    precision, recall, f1 = _prf1(hits, false_alarms, sum(s.misses for s in scores))
    by_kind = []
    for letter in ("F", "E"):
        types = [code for code in categories if code.startswith(letter)]
        counts = [_type_counts(judgment, types) for judgment in judgments]
        by_kind.append(_prf1(*(sum(column) for column in zip(*counts, strict=True)))[2])
    return SubsetMetrics(
        prs=len(scores),
        accuracy=right / len(scores) if scores else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        f1_functional=by_kind[0],
        f1_evolvability=by_kind[1],
        hits=hits,
        false_alarms=false_alarms,
    )


def official_categories() -> list[str]:
    """Give SWR-Bench's category codes, from its scorer.

    Returns:
        list[str]: Codes such as "F.1" and "E.1.1".
    """
    return list(load_evaluation().CHANGE_TYPE_TEXT_MAP)


def check_against_official(judgments: Sequence[dict[str, Any]], ours: SubsetMetrics) -> None:
    """Make sure the metrics computed here match analyze_result()'s on the same PRs.

    Args:
        judgments (Sequence[dict[str, Any]]): The PRs' judgments, with and without issues.
        ours (SubsetMetrics): The metrics computed here.

    Raises:
        RuntimeError: If any differs, which means the formulas here drifted from SWR-Bench's.
    """
    swr = load_evaluation()
    with warnings.catch_warnings():
        # Averages over no points come out as NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        official = swr.analyze_result(copy.deepcopy(list(judgments)))
    theirs = (
        official["overall"]["accuracy"],
        official["overall"]["precision_point"],
        official["overall"]["recall_point"],
        official["overall"]["f1_point"],
        official["change_by_F"]["f1"],
        official["change_by_E"]["f1"],
    )
    mine = (
        ours.accuracy,
        ours.precision,
        ours.recall,
        ours.f1,
        ours.f1_functional,
        ours.f1_evolvability,
    )
    if any(abs(a - b) > 1e-9 for a, b in zip(mine, theirs, strict=True)):
        raise RuntimeError(f"Subset metrics {mine} don't match SWR-Bench's {theirs}")


def load_judgments(folder: Path) -> dict[str, dict[str, Any]]:
    """Read the judge's judgment of every PR of a scored run.

    Args:
        folder (Path): The run's results folder.

    Returns:
        dict[str, dict[str, Any]]: SWR-Bench's evaluate_one() output, by PR.

    Raises:
        DatasetError: If the run isn't scored yet.
    """
    path = folder / JUDGMENTS
    if not path.is_file():
        raise DatasetError(f"{folder.name} isn't scored yet ({path} is missing)")
    details = json.loads(path.read_text(encoding="utf-8"))["details"]
    return {item["instance_id"]: item for item in details if item is not None}


def _row(label: str, metrics: SubsetMetrics) -> str:
    """Write one table row of a run's metrics on a subset.

    Args:
        label (str): The row's name.
        metrics (SubsetMetrics): The metrics.

    Returns:
        str: A Markdown table row.
    """
    return (
        f"| {label} | {metrics.accuracy:.3f} | {metrics.precision:.3f} | "
        f"{metrics.recall:.3f} | {metrics.f1:.3f} | {metrics.f1_functional:.3f} | "
        f"{metrics.f1_evolvability:.3f} | {metrics.hits} | {metrics.false_alarms} |"
    )


def comparison_markdown(
    base_name: str,
    new_name: str,
    base: dict[str, dict[str, Any]],
    new: dict[str, dict[str, Any]],
    with_context: Collection[str],
) -> str:
    """Compare two scored runs on the same PRs, as Markdown.

    Args:
        base_name (str): The run compared against, such as the naive one.
        new_name (str): The run being measured.
        base (dict[str, dict[str, Any]]): The base run's judgments, by PR.
        new (dict[str, dict[str, Any]]): The new run's judgments, by PR.
        with_context (Collection[str]): The PRs whose request in the new run held context.

    Returns:
        str: The text of comparison.md.

    Raises:
        DatasetError: If the runs didn't score the same PRs.
        RuntimeError: If the metrics on all PRs don't match SWR-Bench's own.
    """
    if set(base) != set(new):
        raise DatasetError(f"{base_name} and {new_name} scored different PRs")
    ids = sorted(new)
    held = [i for i in ids if i in with_context]
    others = [i for i in ids if i not in with_context]
    categories = official_categories()
    for judgments in (base, new):
        everything = [judgments[i] for i in ids]
        check_against_official(everything, subset_metrics(everything, categories))
    lines = [
        f"# {new_name} compared with {base_name}",
        "",
        f"The same {len(ids)} PRs, scored by the same judge. Metrics on a subset use "
        "SWR-Bench's formulas over that subset's judgments (no new judge call); on all PRs "
        "they match SWR-Bench's own numbers.",
        "",
        "| PRs | Run | Accuracy | Precision | Recall | F1 | F1 (F) | F1 (E) | Hits | "
        "False alarms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    subsets = [
        (f"All {len(ids)}", ids),
        (f"{len(held)} with context", held),
        (f"{len(others)} without context", others),
    ]
    for label, members in subsets:
        if not members:
            continue
        for name, run in ((base_name, base), (new_name, new)):
            metrics = subset_metrics([run[i] for i in members], categories)
            lines.append(_row(f"{label} | {name}", metrics))
    lines += ["", *pr_changes(base_name, new_name, base, new, ids)]
    return "\n".join(lines) + "\n"


def pr_changes(
    base_name: str,
    new_name: str,
    base: dict[str, dict[str, Any]],
    new: dict[str, dict[str, Any]],
    ids: Sequence[str],
) -> list[str]:
    """Describe what changed PR by PR: hits, false alarms, and verdicts.

    Args:
        base_name (str): The run compared against.
        new_name (str): The run being measured.
        base (dict[str, dict[str, Any]]): The base run's judgments, by PR.
        new (dict[str, dict[str, Any]]): The new run's judgments, by PR.
        ids (Sequence[str]): The PRs.

    Returns:
        list[str]: Markdown lines.
    """
    pairs: list[tuple[PRScore, PRScore]] = [(pr_score(base[i]), pr_score(new[i])) for i in ids]
    gained = [(b, n) for b, n in pairs if n.hits > b.hits]
    lost = [(b, n) for b, n in pairs if n.hits < b.hits]
    clean = [(b, n) for b, n in pairs if not b.has_issues]
    issues = [(b, n) for b, n in pairs if b.has_issues]
    lines = [
        "## PR by PR",
        "",
        f"- Hits: {sum(n.hits for _, n in pairs)} with {new_name}, "
        f"{sum(b.hits for b, _ in pairs)} with {base_name}; "
        f"{len(gained)} PRs gained hits, {len(lost)} lost some",
        f"- False alarms on the {len(clean)} clean PRs: {sum(n.false_alarms for _, n in clean)} "
        f"(was {sum(b.false_alarms for b, _ in clean)}); clean PRs passed: "
        f"{sum(n.judged_good for _, n in clean)} (was {sum(b.judged_good for b, _ in clean)})",
        f"- False alarms on the {len(issues)} PRs with issues: "
        f"{sum(n.false_alarms for _, n in issues)} (was {sum(b.false_alarms for b, _ in issues)})"
        f"; PRs with issues flagged: {sum(not n.judged_good for _, n in issues)} "
        f"(was {sum(not b.judged_good for b, _ in issues)})",
    ]
    changed = [(b, n) for b, n in pairs if (b.hits, b.false_alarms) != (n.hits, n.false_alarms)]
    if changed:
        lines += [
            "",
            f"| PR | Issues | Hits ({base_name} → {new_name}) | False alarms |",
            "|---|---|---|---|",
        ]
        for b, n in changed:
            known = b.hits + b.misses
            lines.append(
                f"| {b.instance_id} | {known} | {b.hits} → {n.hits} | "
                f"{b.false_alarms} → {n.false_alarms} |"
            )
    return lines
