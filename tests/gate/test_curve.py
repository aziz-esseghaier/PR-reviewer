"""Tests for the gate's estimated precision/recall curve (gate/curve.py, gate/chart.py).

Checks that cutting a judgment drops points and turns their hits into misses, that keeping
everything gives back the official scores, and that the curve and its SVG are built.

How to run (from the project folder):
    pytest tests/gate/test_curve.py
"""

from typing import Any

from pr_reviewer.gate.chart import curve_svg, nice_top
from pr_reviewer.gate.curve import (
    confidence_curve,
    cut_judgment,
    evidence_curve,
    policy_point,
    ungated_point,
)
from pr_reviewer.reviewer.compare import subset_metrics
from tests.gate.test_policy import checked, policy

CATEGORIES = ["F.1", "F.4", "E.1"]


def judgment(pr: str, points: int, hits: dict[int, int], issues: int) -> dict[str, Any]:
    """Build an official-style judgment of a PR.

    Args:
        pr (str): The PR.
        points (int): How many review points.
        hits (dict[int, int]): Issue number -> the point number (from 1) that hit it.
        issues (int): How many known issues; 0 for a clean PR.

    Returns:
        dict[str, Any]: The judgment, with every point and issue in category F.4.
    """
    return {
        "instance_id": pr,
        "change_introduced": issues > 0,
        "identified_as_good": "no",
        "pred_points": [
            {"id": f"PRED-POINT-{n}", "change_category": "F.4"} for n in range(1, points + 1)
        ],
        "gt_points": [
            {
                "id": f"GT-{i}",
                "change_category": "F.4",
                "hit": "Yes" if i in hits else "No",
                "hit_by": f"PRED-POINT-{hits[i]}" if i in hits else "",
            }
            for i in range(1, issues + 1)
        ],
    }


def test_cutting_a_judgment_drops_points_and_their_hits() -> None:
    """Check a dropped point's hit becomes a miss, and a kept one stays a hit."""
    full = judgment("a", 2, {1: 1, 2: 2}, 2)
    cut = cut_judgment(full, {"PRED-POINT-2"})
    assert [p["id"] for p in cut["pred_points"]] == ["PRED-POINT-2"]
    assert [(i["hit"], i["hit_by"]) for i in cut["gt_points"]] == [
        ("No", ""),
        ("Yes", "PRED-POINT-2"),
    ]
    assert full["gt_points"][0]["hit"] == "Yes"


def scenario() -> tuple[dict[str, dict[str, Any]], dict[str, list[Any]]]:
    """Build two PRs: one with issues (a tier 1 hit, a tier 4 false alarm) and a clean one.

    Returns:
        tuple[dict[str, dict[str, Any]], dict[str, list[Any]]]: The judgments and the
        checked candidates, by PR.
    """
    judgments = {
        "a": judgment("a", 2, {1: 1}, 2),
        "b": judgment("b", 1, {}, 0),
    }
    found = {
        "a": [checked("f1", 1, 0.6, (1, 1), 0), checked("f2", 4, 0.9, (5, 5), 1)],
        "b": [checked("f1", 3, 0.8, (1, 1), 0)],
    }
    return judgments, found


def test_keeping_everything_gives_the_official_scores() -> None:
    """Check the ungated estimate equals SWR-Bench's formulas on the whole judgments."""
    judgments, found = scenario()
    whole = ungated_point(judgments, found, CATEGORIES)
    official = subset_metrics(list(judgments.values()), CATEGORIES)
    assert (whole.hits, whole.false_alarms, whole.recall) == (
        official.hits,
        official.false_alarms,
        official.recall,
    )
    assert (whole.clean_false_alarms, whole.clean_prs_flagged) == (1, 1)


def test_the_gate_and_the_curves_are_estimated() -> None:
    """Check the tier 1 gate keeps only the hit, and both curves end at everything."""
    judgments, found = scenario()
    estimate = policy_point(judgments, found, policy(), CATEGORIES)
    assert (estimate.comments, estimate.hits, estimate.false_alarms) == (1, 1, 0)
    assert estimate.precision == 1.0 and estimate.recall == 0.5
    by_evidence = evidence_curve(judgments, found, policy(max_per_pr=None), CATEGORIES)
    assert [p.label for p in by_evidence] == [
        "tier <= 1, confidence >= 0.60",
        "tier <= 3, confidence >= 0.80",
        "tier <= 4, confidence >= 0.90",
    ]
    assert [p.comments for p in by_evidence] == [1, 2, 3]
    by_confidence = confidence_curve(judgments, found, policy(max_per_pr=None), CATEGORIES)
    assert [p.comments for p in by_confidence] == [1, 2, 3]
    assert by_confidence[0].hits == 0
    svg = curve_svg(
        by_evidence,
        by_confidence,
        ungated_point(judgments, found, CATEGORIES),
        subset_metrics(list(judgments.values()), CATEGORIES),
    )
    assert svg.startswith("<svg") and svg.count("<circle") >= 6


def test_axis_tops_are_round() -> None:
    """Check axis maximums round up to tidy numbers."""
    assert [nice_top(v) for v in (0.0, 0.21, 0.06, 0.9)] == [0.1, 0.25, 0.1, 1.0]
