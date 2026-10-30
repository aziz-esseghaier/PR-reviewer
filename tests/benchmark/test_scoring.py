"""
Tests for scoring reviews in benchmark/scoring.py.

Runs SWR-Bench's own scorer with a scripted fake judge, so no real provider is called. Checks
reading, writing and lining up the reviews file, the per-PR counts and that they add up to
SWR-Bench's totals, a review with no points, and a PR the judge fails on.

How to run (from the project folder):
    pytest tests/benchmark/test_scoring.py
"""

import json
from pathlib import Path
from typing import Any

import pytest

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.benchmark.judge import JudgeChat
from pr_reviewer.benchmark.scoring import (
    Prediction,
    check_totals,
    load_predictions,
    match_predictions,
    pr_score,
    save_predictions,
    score_reviews,
)
from pr_reviewer.benchmark.source import DatasetError
from tests.benchmark.test_dataset import make_row
from tests.benchmark.test_judge import GOOD_CHANGE, GOOD_CLEAN, with_timeline
from tests.fakes import ScriptedProvider, fake_client

ISSUES = with_timeline(make_row(1, issues=("F.2 Logic", "E.1.1 Textual Changes")))
CLEAN = make_row(2, issues=())
RECORDS = {row["instance_id"]: row for row in (ISSUES, CLEAN)}
PRS = [Instance.model_validate_json(json.dumps(row)) for row in (ISSUES, CLEAN)]

# The judge's reply for ISSUES: one point that hits the first of its two known issues.
ONE_HIT = json.dumps(
    {
        "identified_as_good": "NO",
        "pred_points": [
            {
                "id": "PRED-POINT-1",
                "description": "x",
                "change_category": "F.2",
                "severity_score": 7,
            }
        ],
        "gt_points": [
            {
                "id": "GT-POINT-1",
                "description": "a",
                "change_category": "F.2",
                "hit": "YES",
                "hit_by": "PRED-POINT-1",
            },
            {
                "id": "GT-POINT-2",
                "description": "b",
                "change_category": "E.1.1",
                "hit": "NO",
                "hit_by": "N/A",
            },
        ],
    }
)
# The judge's reply for a review with no points on ISSUES.
NO_POINTS = json.dumps(
    {
        "identified_as_good": "YES",
        "pred_points": [],
        "gt_points": [
            {
                "id": "GT-POINT-1",
                "description": "a",
                "change_category": "F.2",
                "hit": "NO",
                "hit_by": "N/A",
            },
            {
                "id": "GT-POINT-2",
                "description": "b",
                "change_category": "E.1.1",
                "hit": "NO",
                "hit_by": "N/A",
            },
        ],
    }
)
# The judge's reply for CLEAN when the review raises one point: a false alarm.
ONE_FALSE_ALARM = json.dumps(
    {
        "identified_as_good": "NO",
        "pred_points": [
            {
                "id": "PRED-POINT-1",
                "description": "y",
                "change_category": "E.2",
                "severity_score": 2,
            }
        ],
    }
)


def write_predictions(folder: Path, lines: list[str]) -> Path:
    """Write a predictions file.

    Args:
        folder (Path): Folder to write into.
        lines (list[str]): The file's lines.

    Returns:
        Path: The file.
    """
    path = folder / "reviews.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def reviews(*texts: str) -> list[Prediction]:
    """Make one review per PR of PRS.

    Args:
        *texts (str): The review texts, in the order of PRS.

    Returns:
        list[Prediction]: The reviews.
    """
    pairs = zip(PRS, texts, strict=True)
    return [Prediction(instance_id=pr.instance_id, review=text) for pr, text in pairs]


def run_with(replies: list[str | int], *texts: str) -> Any:
    """Score reviews of PRS with a judge that gives scripted replies.

    Args:
        replies (list[str | int]): The judge's replies in order; a number is an HTTP error.
        *texts (str): The review texts, in the order of PRS.

    Returns:
        Any: The ScoreRun.
    """
    chat = JudgeChat(fake_client(ScriptedProvider(replies)))
    return score_reviews(chat, RECORDS, reviews(*texts))


def test_predictions_file_is_read_line_by_line(tmp_path: Path) -> None:
    """Check that reviews are read in order, with extra keys such as a model name ignored."""
    path = write_predictions(
        tmp_path,
        [
            '{"instance_id": "psf__requests-2", "review": "Fine.", "model": "x"}',
            "",
            '{"instance_id": "psf__requests-1", "review": "1. Bug."}',
        ],
    )
    assert [p.instance_id for p in load_predictions(path)] == ["psf__requests-2", "psf__requests-1"]


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (['{"instance_id": "a"}'], "line 1: review: Field required"),
        (['{"instance_id": "a", "review": "x"}', "not json"], "line 2"),
        (
            ['{"instance_id": "a", "review": "x"}', '{"instance_id": "a", "review": "y"}'],
            "line 2: a has a second review",
        ),
    ],
)
def test_bad_predictions_file_names_the_line(
    tmp_path: Path, lines: list[str], message: str
) -> None:
    """Check that a broken line or a second review of a PR is refused, naming the line."""
    with pytest.raises(DatasetError, match=message):
        load_predictions(write_predictions(tmp_path, lines))


def test_saved_predictions_read_back_unchanged(tmp_path: Path) -> None:
    """Check that a written file reads back the same, and writing again gives the same bytes."""
    written = reviews("1. Uses “smart” quotes.\n2. Naïve loop.", "Fine.")
    path = tmp_path / "out" / "reviews.jsonl"
    save_predictions(path, written)
    first = path.read_bytes()
    save_predictions(path, written)
    assert path.read_bytes() == first
    assert b"\r\n" not in first
    assert load_predictions(path) == written


def test_reviews_are_lined_up_with_the_set() -> None:
    """Check that reviews come back in the set's order, and gaps or extras are refused."""
    clean, issues = reviews("a", "b")[::-1]
    assert [p.instance_id for p in match_predictions([clean, issues], PRS, "dev")] == [
        "psf__requests-1",
        "psf__requests-2",
    ]
    with pytest.raises(DatasetError, match="1 of the 2 dev PRs have no review, such as psf__"):
        match_predictions([issues], PRS, "dev")
    extra = Prediction(instance_id="psf__requests-9", review="x")
    with pytest.raises(DatasetError, match="1 reviews are for PRs outside the dev set"):
        match_predictions([issues, clean, extra], PRS, "dev")


def test_one_point_hitting_two_issues_is_one_hit() -> None:
    """Check SWR-Bench's rule: a point counts once, even when it matches two known issues."""
    result = json.loads(NO_POINTS) | {"instance_id": "x", "change_introduced": True}
    result["identified_as_good"] = "NO"
    result["pred_points"] = json.loads(ONE_HIT)["pred_points"]
    for issue in result["gt_points"]:
        issue.update(hit="YES", hit_by="PRED-POINT-1")
    score = pr_score(result)
    assert (score.hits, score.false_alarms, score.misses) == (1, 0, 1)
    assert not score.judged_good


def test_approving_review_passes_the_pr_even_with_points() -> None:
    """Check SWR-Bench's rule: a review the judge reads as approving passes the PR.

    Such as "LGTM, one small nit": the nit is still a point, but the PR counts as passed.
    """
    result = json.loads(ONE_FALSE_ALARM) | {"instance_id": "x", "change_introduced": False}
    result["identified_as_good"] = "YES"
    score = pr_score(result)
    assert score.judged_good
    assert score.false_alarms == 1


def test_scores_add_up_to_swr_bench_totals() -> None:
    """Check a full run: a hit and a miss on the PR with issues, a false alarm on the clean PR."""
    run = run_with([ONE_HIT, ONE_FALSE_ALARM], "1. Divides by zero.", "1. Odd spacing.")
    assert run.failed == []
    assert [(s.points, s.hits, s.false_alarms, s.misses) for s in run.scores] == [
        (1, 1, 0, 1),
        (1, 0, 1, 0),
    ]
    overall = run.analysis["overall"]
    assert (overall["tp_point"], overall["fp_point"], overall["fn_point"]) == (1, 1, 1)
    assert (overall["tp"], overall["fp"], overall["fn"], overall["tn"]) == (1, 1, 0, 0)
    assert run.analysis["change_by_F"]["tp"] == 1


def test_review_with_no_points_passes_every_pr() -> None:
    """Check the no-comments case: clean PRs pass, PRs with issues are missed, no NaN crash."""
    run = run_with([NO_POINTS, GOOD_CLEAN], "No issues found.", "No issues found.")
    overall = run.analysis["overall"]
    assert (overall["tp"], overall["fn"], overall["fp"], overall["tn"]) == (0, 1, 0, 1)
    assert (overall["recall_point"], overall["precision_point"]) == (0, 0)


def test_failed_pr_is_reported_and_leaves_no_metrics() -> None:
    """Check that a PR without a valid judgment after 3 tries fails the run, not silently."""
    run = run_with(["bad", "bad", "bad", GOOD_CLEAN], "1. Bug.", "Fine.")
    assert run.failed == ["psf__requests-1"]
    assert run.analysis is None
    assert len(run.scores) == 1


def test_counts_that_disagree_with_swr_bench_are_caught() -> None:
    """Check that check_totals() notices per-PR counts that don't match SWR-Bench's."""
    run = run_with([ONE_HIT, ONE_FALSE_ALARM], "1. Divides by zero.", "1. Odd spacing.")
    run.analysis["overall"]["fp_point"] += 1
    with pytest.raises(RuntimeError, match="don't add up"):
        check_totals(run.scores, run.analysis)


def test_good_change_reply_from_the_judge_tests_is_one_full_hit() -> None:
    """Check the shared GOOD_CHANGE reply against a PR with one known issue."""
    result = json.loads(GOOD_CHANGE) | {"instance_id": "x", "change_introduced": True}
    assert pr_score(result).hits == 1
