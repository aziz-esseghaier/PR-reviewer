"""Where scores are kept: a folder under results/, and three tables in PostgreSQL.

results/<name>/ holds one scored set of reviews:
- config.json: what was scored and how (reviews file fingerprint, set, reviewer, judge,
  SWR-Bench version). Kept in git.
- metrics.json: SWR-Bench's metrics, as its analyze_result() gives them. Kept in git.
- summary.md: the main metrics as readable tables. Kept in git.
- raw/: the reviews file and SWR-Bench's full output, with each judge reply. Not in git.
When our own reviewer wrote the reviews, the same folder also holds its run: review_config.json
and review_summary.md, with the reviews and findings in raw/ (see reviewer/commands.py).

The database gets the benchmark PRs (bench_prs), the experiment (experiments) and one row of
counts per PR (pr_scores), so later results can be grouped by repo, year or kind with SQL.

Numbers SWR-Bench leaves as NaN, such as the average severity of no points, are saved as null,
so the JSON files stay valid JSON.

Imported by other modules; the file isn't run directly.
"""

import json
import math
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from pr_reviewer.benchmark.dataset import Instance, select_prs
from pr_reviewer.benchmark.scoring import ScoreRun
from pr_reviewer.benchmark.source import DATASET, SOURCE_COMMIT, SOURCE_REPO
from pr_reviewer.benchmark.split import SPLIT_SHA256, Split
from pr_reviewer.core.config import ConfigError, project_root

# Experiment names become folder names, so they stay short and simple.
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

# The settings file of a review run (reviewer/commands.py), in the folder its scores go to.
REVIEW_CONFIG = "review_config.json"


def results_folder(name: str) -> Path:
    """Give the folder of an experiment, after checking its name.

    Args:
        name (str): The experiment name, such as "no-comments-baseline".

    Returns:
        Path: results/<name> in the project folder.

    Raises:
        ConfigError: If the name isn't lowercase letters, digits, ".", "_" and "-", starting
            with a letter or digit, at most 64 characters.
    """
    if not NAME_PATTERN.fullmatch(name):
        raise ConfigError(
            f"Bad experiment name {name!r}: use lowercase letters, digits, '.', '_' and '-', "
            "such as no-comments-baseline"
        )
    return project_root() / "results" / name


def check_folder(folder: Path, predictions_sha256: str) -> None:
    """Make sure a results folder is free, or holds the scores of the same reviews file.

    Scoring the same reviews again under the same name replaces the old results, which is how
    a stopped run is finished. Another reviews file needs another name. A folder that holds
    only a review run, not scored yet, counts as free: its scores go next to it.

    Args:
        folder (Path): The experiment's folder.
        predictions_sha256 (str): Fingerprint of the reviews file about to be scored.

    Raises:
        ConfigError: If the folder holds something else.
    """
    if not folder.exists() or not any(folder.iterdir()):
        return
    config = folder / "config.json"
    if not config.is_file() and (folder / REVIEW_CONFIG).is_file():
        return
    saved = json.loads(config.read_text(encoding="utf-8")) if config.is_file() else {}
    if saved.get("predictions", {}).get("sha256") != predictions_sha256:
        raise ConfigError(
            f"{folder} already holds other results. Pick another --name, or delete the folder "
            "to reuse the name."
        )


@dataclass(frozen=True)
class Experiment:
    """What was scored and how: the contents of config.json.

    Attributes:
        name (str): The experiment name.
        date (str): When it was scored, as YYYY-MM-DD.
        split (str): "dev" or "test".
        reviewer (str): The reviewer's model, or "none" for a baseline without one.
        predictions_file (str): The reviews file's name, without its folders.
        predictions_sha256 (str): The reviews file's fingerprint.
        reviews (int): How many reviews were scored.
        judge (dict[str, Any]): The judge's profile, provider, model, temperature and max_tokens.
        since (int | None): Only the set's PRs from this year on were scored. None means all.
    """

    name: str
    date: str
    split: str
    reviewer: str
    predictions_file: str
    predictions_sha256: str
    reviews: int
    judge: dict[str, Any]
    since: int | None = None

    def config(self) -> dict[str, Any]:
        """Give the experiment as the JSON object saved in config.json.

        Returns:
            dict[str, Any]: The config. "since" appears only when a subset was scored.
        """
        subset = {} if self.since is None else {"since": self.since}
        return {
            "name": self.name,
            "date": self.date,
            "split": self.split,
            **subset,
            "reviewer": self.reviewer,
            "predictions": {
                "file": self.predictions_file,
                "sha256": self.predictions_sha256,
                "reviews": self.reviews,
            },
            "judge": self.judge,
            "swr_bench": {
                "repo": SOURCE_REPO,
                "commit": SOURCE_COMMIT,
                "dataset_sha256": DATASET.sha256,
                "split_sha256": SPLIT_SHA256,
            },
        }


def plain_json(value: Any) -> Any:
    """Turn SWR-Bench's numbers into plain JSON values: numpy types to Python, NaN to None.

    Args:
        value (Any): A number, or a dict or list holding numbers.

    Returns:
        Any: The same structure with plain values.
    """
    if isinstance(value, dict):
        return {key: plain_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _number(value: Any) -> str:
    """Write a metric with 3 decimals, or "n/a" for one SWR-Bench left as NaN.

    Args:
        value (Any): The metric.

    Returns:
        str: Such as "0.412".
    """
    return "n/a" if value is None else f"{value:.3f}"


def summary_markdown(experiment: Experiment, analysis: dict[str, Any], prs: int) -> str:
    """Write the main metrics as Markdown tables.

    Args:
        experiment (Experiment): What was scored.
        analysis (dict[str, Any]): SWR-Bench's metrics, as plain JSON values.
        prs (int): How many PRs were scored.

    Returns:
        str: The text of summary.md.
    """
    overall, change, clean = analysis["overall"], analysis["change"], analysis["clean"]
    rows = [
        (
            "All",
            overall["precision_point"],
            overall["recall_point"],
            overall["f1_point"],
            overall["tp_point"],
            overall["fp_point"],
            overall["fn_point"],
        )
    ]
    for label, key in (("Functional (F)", "change_by_F"), ("Evolvability (E)", "change_by_E")):
        part = analysis[key]
        rows.append(
            (
                label,
                part["precision"],
                part["recall"],
                part["f1"],
                part["tp"],
                part["fp"],
                part["fn"],
            )
        )
    judge = experiment.judge
    subset = "" if experiment.since is None else f" from {experiment.since} on"
    lines = [
        f"# Scores: {experiment.name}",
        "",
        f"- Reviews: `{experiment.predictions_file}`, {prs} PRs of the {experiment.split} set"
        f"{subset} ({change['count']} with known issues, {clean['count']} clean)",
        f"- Reviewer: {experiment.reviewer}",
        f"- Judge: {judge['model']} (temperature {judge['temperature']}, replies up to "
        f"{judge['max_tokens']:,} tokens), SWR-Bench's scorer at {SOURCE_COMMIT[:7]}",
        f"- Scored: {experiment.date}",
        "",
        "## PR level: does the review flag PRs with issues and pass clean ones?",
        "",
        "| Accuracy | Precision | Recall | F1 | Flagged, with issues | Missed, with issues "
        "| False alarm, clean | Passed, clean |",
        "|---|---|---|---|---|---|---|---|",
        f"| {_number(overall['accuracy'])} | {_number(overall['precision'])} "
        f"| {_number(overall['recall'])} | {_number(overall['f1'])} | {overall['tp']} "
        f"| {overall['fn']} | {overall['fp']} | {overall['tn']} |",
        "",
        "A review passes a PR when the judge reads it as approving, or finds no point in it.",
        "",
        "## Point level: do the review's points match the known issues?",
        "",
        "| Issues | Precision | Recall | F1 | Hits | False alarms | Misses |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {label} | {_number(p)} | {_number(r)} | {_number(f)} | {tp} | {fp} | {fn} |"
        for label, p, r, f, tp, fp, fn in rows
    ]
    lines += [
        "",
        "A point is a hit when it matches a known issue and a false alarm when it matches none "
        "(every point on a clean PR is one). A known issue no point matches is a miss. One "
        "point matching several issues counts as one hit, the rest as misses (SWR-Bench's "
        "rule). The F and E rows sort points by the category the judge gave them.",
        "",
        f"Review points per PR: {overall['avg_pred_points']:.2f}.",
    ]
    if overall["tp_point"] + overall["fp_point"] == 0:
        lines += [
            "",
            "No review raised a point, so precision has nothing to measure: SWR-Bench reports 0.",
        ]
    return "\n".join(lines) + "\n"


def write_results(
    folder: Path, experiment: Experiment, run: ScoreRun, predictions_path: Path
) -> None:
    """Write an experiment's folder: config, metrics, summary and raw outputs.

    Args:
        folder (Path): The experiment's folder. Created if missing.
        experiment (Experiment): What was scored.
        run (ScoreRun): The finished run. Must have metrics, so no PR failed.
        predictions_path (Path): The reviews file, copied into raw/.

    Raises:
        ValueError: If the run has no metrics.
    """
    if run.analysis is None:
        raise ValueError("A run with failed PRs has no metrics to write")
    analysis = plain_json(run.analysis)
    raw = folder / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    def write(path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")

    write(folder / "config.json", experiment.config())
    write(folder / "metrics.json", analysis)
    summary = summary_markdown(experiment, analysis, len(run.instance_ids))
    (folder / "summary.md").write_text(summary, encoding="utf-8", newline="\n")
    shutil.copyfile(predictions_path, raw / "predictions.jsonl")
    # The same shape SWR-Bench's own runner saves.
    write(raw / "swrbench_output.json", {"analysis_results": analysis, "details": run.results})


def save_bench_prs(
    conn: psycopg.Connection[TupleRow], instances: Sequence[Instance], split: Split
) -> None:
    """Write the benchmark PRs, with their kind and set, to bench_prs.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        instances (Sequence[Instance]): All PRs from the dataset.
        split (Split): The dev/test split.
    """
    rows = [
        (
            pr.instance_id,
            pr.repo,
            pr.year,
            pr.group,
            split.which(pr.instance_id),
            sum(issue.is_functional for issue in pr.changes),
            sum(not issue.is_functional for issue in pr.changes),
        )
        for pr in select_prs(instances)
    ]
    with conn.transaction(), conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO bench_prs (instance_id, repo, year, kind, split, functional_issues,
                                   evolvability_issues)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (instance_id) DO UPDATE SET
                repo = EXCLUDED.repo, year = EXCLUDED.year, kind = EXCLUDED.kind,
                split = EXCLUDED.split, functional_issues = EXCLUDED.functional_issues,
                evolvability_issues = EXCLUDED.evolvability_issues
            """,
            rows,
        )


def save_scores(conn: psycopg.Connection[TupleRow], experiment: Experiment, run: ScoreRun) -> None:
    """Write an experiment and its per-PR counts, replacing an earlier run of the same name.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection. bench_prs must hold the PRs.
        experiment (Experiment): What was scored.
        run (ScoreRun): The finished run.
    """
    judgments = {result["instance_id"]: result for result in run.results if result is not None}
    with conn.transaction(), conn.cursor() as cursor:
        cursor.execute("DELETE FROM experiments WHERE name = %s", (experiment.name,))
        cursor.execute(
            """
            INSERT INTO experiments (name, split, reviewer, judge_model, config)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                experiment.name,
                experiment.split,
                experiment.reviewer,
                experiment.judge["model"],
                Jsonb(experiment.config()),
            ),
        )
        cursor.executemany(
            """
            INSERT INTO pr_scores (experiment, instance_id, judged_good, points, hits,
                                   false_alarms, misses, judgment)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    experiment.name,
                    score.instance_id,
                    score.judged_good,
                    score.points,
                    score.hits,
                    score.false_alarms,
                    score.misses,
                    Jsonb(
                        {
                            key: judgments[score.instance_id].get(key)
                            for key in ("identified_as_good", "pred_points", "gt_points")
                        }
                    ),
                )
                for score in run.scores
            ],
        )
