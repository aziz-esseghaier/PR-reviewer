"""Scoring a set of reviews with SWR-Bench's official scorer and the project's judge.

A predictions file is JSONL with one review per PR, {"instance_id": ..., "review": ...}, the
format SWR-Bench's own review generator writes (load_predictions() reads one,
save_predictions() writes one). score_reviews() runs SWR-Bench's
evaluate_one() on each PR, with the judge answering through our LLM client (benchmark/judge.py),
then its analyze_result() for the metrics. Prompts, answer checks and formulas are all
SWR-Bench's.

SWR-Bench's own runner silently drops a PR whose judging failed. Here a failed PR is kept as
None and reported, and a run gets metrics only when every PR was judged.

pr_score() counts one PR the way analyze_result() counts it, so scores can be stored per PR.
Each run checks that these counts add up to SWR-Bench's totals.

Imported by other modules; the file isn't run directly.
"""

import copy
import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.benchmark.judge import JudgeChat, scorer_settings
from pr_reviewer.benchmark.official import load_evaluation
from pr_reviewer.benchmark.source import DatasetError

# SWR-Bench's scorer logs its warnings about odd judge replies here.
SCORER_LOG = logging.getLogger("pr_reviewer.swrbench")

# Called after each PR: (position, instance ID, SWR-Bench's judgment or None if it failed).
ProgressCallback = Callable[[int, str, dict[str, Any] | None], None]


class Prediction(BaseModel):
    """One review to score.

    Attributes:
        instance_id (str): The PR it reviews.
        review (str): The review text, as the reviewer would post it.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    instance_id: str
    review: str


def load_predictions(path: Path) -> list[Prediction]:
    """Read a predictions file: one JSON object per line.

    Args:
        path (Path): The JSONL file.

    Returns:
        list[Prediction]: The reviews, in file order.

    Raises:
        DatasetError: If the file is missing, a line isn't a review, or a PR has two reviews.
    """
    if not path.is_file():
        raise DatasetError(f"{path} not found")
    predictions: list[Prediction] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                prediction = Prediction.model_validate_json(line)
            except ValidationError as exc:
                error = exc.errors()[0]
                where = ".".join(str(part) for part in error["loc"]) or "(line)"
                raise DatasetError(f"{path.name}, line {number}: {where}: {error['msg']}") from None
            if prediction.instance_id in seen:
                raise DatasetError(
                    f"{path.name}, line {number}: {prediction.instance_id} has a second review"
                )
            seen.add(prediction.instance_id)
            predictions.append(prediction)
    return predictions


def save_predictions(path: Path, predictions: Sequence[Prediction]) -> None:
    """Write reviews as a predictions file that load_predictions() reads.

    The same reviews always give the same bytes, so the file's fingerprint stays the same when
    it's written again.

    Args:
        path (Path): The JSONL file to write. Its folder is created if needed, and an existing
            file is replaced.
        predictions (Sequence[Prediction]): The reviews, one per PR, in the order to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for prediction in predictions:
            file.write(prediction.model_dump_json() + "\n")


def match_predictions(
    predictions: Sequence[Prediction], prs: Sequence[Instance], set_name: str
) -> list[Prediction]:
    """Line up exactly one review per PR of a set, in the set's order.

    Args:
        predictions (Sequence[Prediction]): The reviews read from the file.
        prs (Sequence[Instance]): The PRs of the set being scored.
        set_name (str): "dev" or "test", for messages.

    Returns:
        list[Prediction]: One review per PR, in the order of `prs`.

    Raises:
        DatasetError: If a PR of the set has no review, or a review is for another PR.
    """
    by_id = {prediction.instance_id: prediction for prediction in predictions}
    wanted = [pr.instance_id for pr in prs]
    missing = [instance_id for instance_id in wanted if instance_id not in by_id]
    if missing:
        raise DatasetError(
            f"{len(missing)} of the {len(wanted)} {set_name} PRs have no review, such as "
            f"{missing[0]}. Every PR needs one; a PR with nothing to report gets a review "
            "saying so, such as 'The pull request looks good and meets the standards.'"
        )
    extra = sorted(by_id.keys() - set(wanted))
    if extra:
        raise DatasetError(
            f"{len(extra)} reviews are for PRs outside the {set_name} set, such as {extra[0]}."
        )
    return [by_id[instance_id] for instance_id in wanted]


@dataclass(frozen=True)
class PRScore:
    """One PR's counts, the way SWR-Bench's analyze_result() counts them.

    Attributes:
        instance_id (str): The PR.
        has_issues (bool): Whether the PR has known issues.
        judged_good (bool): Whether the review passed the PR: the judge read it as approving,
            or found no point in it.
        points (int): Review points the judge found.
        hits (int): Points that matched a known issue. A point that matches several issues
            counts once.
        false_alarms (int): Points that matched no known issue. On a clean PR, every point.
        misses (int): Known issues minus hits.
    """

    instance_id: str
    has_issues: bool
    judged_good: bool
    points: int
    hits: int
    false_alarms: int
    misses: int


def pr_score(result: dict[str, Any]) -> PRScore:
    """Count one PR's judgment as SWR-Bench's analyze_result() does.

    Args:
        result (dict[str, Any]): SWR-Bench's evaluate_one() output for the PR.

    Returns:
        PRScore: The PR's counts.
    """
    points = result["pred_points"]
    judged_good = result["identified_as_good"].lower() == "yes" or not points
    if not result["change_introduced"]:
        return PRScore(result["instance_id"], False, judged_good, len(points), 0, len(points), 0)
    point_ids = {point["id"] for point in points}
    hit_ids = {issue["hit_by"] for issue in result["gt_points"]} & point_ids
    return PRScore(
        instance_id=result["instance_id"],
        has_issues=True,
        judged_good=judged_good,
        points=len(points),
        hits=len(hit_ids),
        false_alarms=len(point_ids - hit_ids),
        misses=len(result["gt_points"]) - len(hit_ids),
    )


@dataclass
class ScoreRun:
    """The outcome of scoring one set of reviews.

    Attributes:
        instance_ids (list[str]): The PRs scored, in order.
        results (list[dict[str, Any] | None]): SWR-Bench's judgment of each PR, from
            evaluate_one(). None where judging failed.
        analysis (dict[str, Any] | None): SWR-Bench's metrics, from analyze_result(). None
            when a PR failed, since metrics over part of a set would mislead.
        scores (list[PRScore]): The counts of each judged PR.
    """

    instance_ids: list[str]
    results: list[dict[str, Any] | None]
    analysis: dict[str, Any] | None
    scores: list[PRScore]

    @property
    def failed(self) -> list[str]:
        """The PRs whose judging failed."""
        pairs = zip(self.instance_ids, self.results, strict=True)
        return [instance_id for instance_id, result in pairs if result is None]


def check_totals(scores: Sequence[PRScore], analysis: dict[str, Any]) -> None:
    """Make sure the per-PR counts add up to SWR-Bench's own totals.

    Args:
        scores (Sequence[PRScore]): The counts of every PR.
        analysis (dict[str, Any]): SWR-Bench's metrics for the same PRs.

    Raises:
        RuntimeError: If any total differs, which means pr_score() no longer counts the way
            SWR-Bench does.
    """
    overall = analysis["overall"]
    ours = {
        "tp_point": sum(score.hits for score in scores),
        "fp_point": sum(score.false_alarms for score in scores),
        "fn_point": sum(score.misses for score in scores),
        "tp": sum(score.has_issues and not score.judged_good for score in scores),
        "fn": sum(score.has_issues and score.judged_good for score in scores),
        "fp": sum(not score.has_issues and not score.judged_good for score in scores),
        "tn": sum(not score.has_issues and score.judged_good for score in scores),
    }
    theirs = {key: overall[key] for key in ours}
    if ours != theirs:
        raise RuntimeError(f"Per-PR counts {ours} don't add up to SWR-Bench's totals {theirs}")


def score_reviews(
    chat: JudgeChat,
    records: dict[str, dict[str, Any]],
    predictions: Sequence[Prediction],
    on_result: ProgressCallback | None = None,
) -> ScoreRun:
    """Score each review with SWR-Bench's scorer, then compute SWR-Bench's metrics.

    Args:
        chat (JudgeChat): The judge, answering SWR-Bench's requests through our client.
        records (dict[str, dict[str, Any]]): The raw dataset record of each PR.
        predictions (Sequence[Prediction]): One review per PR, from match_predictions().
        on_result (ProgressCallback | None): Called after each PR, to show progress.

    Returns:
        ScoreRun: Each PR's judgment and counts, and the metrics if every PR was judged.

    Raises:
        LimitError: If the judge's daily quota runs out. The replies so far are cached, so
            running again picks up where this stopped.
    """
    swr = load_evaluation(chat)
    settings = scorer_settings(chat.client)
    results: list[dict[str, Any] | None] = []
    for position, prediction in enumerate(predictions):
        item = {
            "instance": copy.deepcopy(records[prediction.instance_id]),
            "pred": {"instance_id": prediction.instance_id, "review": prediction.review},
        }
        result: dict[str, Any] | None = swr.evaluate_one(settings, item, SCORER_LOG)
        results.append(result)
        if on_result is not None:
            on_result(position, prediction.instance_id, result)
    scores = [pr_score(result) for result in results if result is not None]
    run = ScoreRun([p.instance_id for p in predictions], results, None, scores)
    if run.failed:
        return run
    with warnings.catch_warnings():
        # Averages over no points, such as the severity of an empty review, come out as NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        run.analysis = swr.analyze_result(results)
    check_totals(scores, run.analysis)
    return run
