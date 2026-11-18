"""The precision/recall curve of the gate, estimated from a scored ungated run.

Scoring a review costs one judge call per PR, so scoring the gate at every cutoff would cost
many days of the judge's free quota. Instead, the ungated review (every candidate) is scored
once, and its judgment says, point by point, which points hit a known issue. A stricter
cutoff keeps a subset of those points, so its judgment is estimated by cutting the ungated one
down: the dropped points go, and a known issue stays hit only if a point that hit it is kept.
SWR-Bench's own formulas (reviewer/compare.py) then give precision, recall and F1.

This is an estimate: the judge saw all the points together, and might judge a shorter review a
little differently. The real gate is also scored officially, so the two can be compared.

The judge numbered the review's points in order, so point k of a PR is its candidate k
(evidence/report.py relies on the same). A PR whose point count doesn't match its candidates
can't be split: its points are kept only at a cutoff that keeps all its candidates.

Two curves:
- "evidence": findings sorted by tier, then confidence (the gate's own order);
- "confidence": by the reviewer's confidence alone, the comparison without evidence.
Both apply the gate's one-comment-per-place rule and cap per PR.

Imported by other modules; the file isn't run directly.
"""

import copy
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pr_reviewer.gate.policy import Checked, GatePolicy, gate, rank, take
from pr_reviewer.reviewer.compare import kind_scores, subset_metrics


@dataclass(frozen=True)
class CurvePoint:
    """The estimated scores at one cutoff.

    Attributes:
        label (str): The cutoff, such as "tier <= 1, confidence >= 0.8".
        comments (int): Comments kept on all PRs.
        hits (int): Kept points that hit a known issue.
        false_alarms (int): Kept points that hit none, on PRs with issues or clean ones.
        clean_false_alarms (int): Kept points on clean PRs.
        clean_prs_flagged (int): Clean PRs with at least one kept point.
        precision (float): Point-level precision.
        recall (float): Point-level recall.
        f1 (float): Point-level F1.
        recall_functional (float): Recall over functional (F) issues.
        precision_functional (float): Precision over functional (F) points.
        f1_functional (float): F1 over functional points and issues.
    """

    label: str
    comments: int
    hits: int
    false_alarms: int
    clean_false_alarms: int
    clean_prs_flagged: int
    precision: float
    recall: float
    f1: float
    recall_functional: float
    precision_functional: float
    f1_functional: float


def cut_judgment(judgment: Mapping[str, Any], kept: Collection[str]) -> dict[str, Any]:
    """Cut a PR's judgment down to some of its points.

    Args:
        judgment (Mapping[str, Any]): SWR-Bench's judgment of the ungated review.
        kept (Collection[str]): The point IDs kept, such as {"PRED-POINT-1"}.

    Returns:
        dict[str, Any]: A copy with only the kept points; a known issue hit only by dropped
        points becomes a miss.
    """
    cut = copy.deepcopy(dict(judgment))
    cut["pred_points"] = [p for p in cut.get("pred_points", []) if p.get("id") in kept]
    for issue in cut.get("gt_points", []):
        if str(issue.get("hit", "")).lower() == "yes" and issue.get("hit_by") not in kept:
            issue["hit"] = "No"
            issue["hit_by"] = ""
    return cut


def kept_points(
    judgment: Mapping[str, Any], found: Sequence[Checked], taken: Sequence[Checked]
) -> set[str]:
    """Give the IDs of the points a PR keeps, from the candidates taken.

    Args:
        judgment (Mapping[str, Any]): The PR's ungated judgment.
        found (Sequence[Checked]): All its candidates, in the reviewer's order.
        taken (Sequence[Checked]): The ones a cutoff keeps.

    Returns:
        set[str]: Point IDs. When the points don't line up with the candidates, all of them
        if every candidate is taken, else none.
    """
    points = judgment.get("pred_points") or []
    if len(points) != len(found):
        return {p.get("id") for p in points} if len(taken) == len(found) and found else set()
    orders = {item.order for item in taken}
    return {point.get("id") for number, point in enumerate(points) if number in orders}


def score_cutoff(
    label: str,
    judgments: Mapping[str, Mapping[str, Any]],
    checked: Mapping[str, Sequence[Checked]],
    choose: Callable[[Sequence[Checked]], list[Checked]],
    categories: Collection[str],
) -> CurvePoint:
    """Estimate the scores of one cutoff over every PR.

    Args:
        label (str): The cutoff's name.
        judgments (Mapping[str, Mapping[str, Any]]): The ungated judgments, by PR.
        checked (Mapping[str, Sequence[Checked]]): The checked candidates, by PR (PRs without
            candidates may be missing).
        choose (Callable[[Sequence[Checked]], list[Checked]]): Picks a PR's kept findings.
        categories (Collection[str]): Every SWR-Bench category code.

    Returns:
        CurvePoint: The estimated scores.
    """
    cut = []
    comments = 0
    for pr, judgment in judgments.items():
        found = checked.get(pr, [])
        taken = choose(found)
        comments += len(taken)
        cut.append(cut_judgment(judgment, kept_points(judgment, found, taken)))
    metrics = subset_metrics(cut, categories)
    clean = [j for j in cut if not j["change_introduced"]]
    precision_f, recall_f, f1_f = kind_scores(cut, categories, "F")
    return CurvePoint(
        label=label,
        comments=comments,
        hits=metrics.hits,
        false_alarms=metrics.false_alarms,
        clean_false_alarms=sum(len(j["pred_points"]) for j in clean),
        clean_prs_flagged=sum(bool(j["pred_points"]) for j in clean),
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        recall_functional=recall_f,
        precision_functional=precision_f,
        f1_functional=f1_f,
    )


def evidence_curve(
    judgments: Mapping[str, Mapping[str, Any]],
    checked: Mapping[str, Sequence[Checked]],
    policy: GatePolicy,
    categories: Collection[str],
) -> list[CurvePoint]:
    """Score every cutoff of the gate's order: tier, then confidence.

    Args:
        judgments (Mapping[str, Mapping[str, Any]]): The ungated judgments, by PR.
        checked (Mapping[str, Sequence[Checked]]): The checked candidates, by PR.
        policy (GatePolicy): The gate's settings; its dedupe and cap apply at every cutoff.
        categories (Collection[str]): Every SWR-Bench category code.

    Returns:
        list[CurvePoint]: One point per (tier, confidence) cutoff, strictest first.
    """
    cutoffs = sorted({(c.tier, c.confidence) for found in checked.values() for c in found})
    cutoffs.sort(key=lambda cut: (cut[0], -cut[1]))
    points = []
    for tier, confidence in cutoffs:

        def keep(item: Checked, tier: int = tier, confidence: float = confidence) -> bool:
            return (item.tier, -item.confidence) <= (tier, -confidence)

        def choose(
            found: Sequence[Checked], keep: Callable[[Checked], bool] = keep
        ) -> list[Checked]:
            return take(found, keep, rank, policy.dedupe, policy.max_per_pr)

        label = f"tier <= {tier}, confidence >= {confidence:.2f}"
        points.append(score_cutoff(label, judgments, checked, choose, categories))
    return points


def confidence_curve(
    judgments: Mapping[str, Mapping[str, Any]],
    checked: Mapping[str, Sequence[Checked]],
    policy: GatePolicy,
    categories: Collection[str],
) -> list[CurvePoint]:
    """Score every cutoff of the reviewer's confidence alone, ignoring the evidence.

    Args:
        judgments (Mapping[str, Mapping[str, Any]]): The ungated judgments, by PR.
        checked (Mapping[str, Sequence[Checked]]): The checked candidates, by PR.
        policy (GatePolicy): The gate's settings; its dedupe and cap apply at every cutoff.
        categories (Collection[str]): Every SWR-Bench category code.

    Returns:
        list[CurvePoint]: One point per confidence value, strictest first.
    """
    values = sorted({c.confidence for found in checked.values() for c in found}, reverse=True)
    points = []
    for value in values:

        def choose(found: Sequence[Checked], value: float = value) -> list[Checked]:
            return take(
                found,
                lambda item: item.confidence >= value,
                lambda item: (-item.confidence, item.order),
                policy.dedupe,
                policy.max_per_pr,
            )

        label = f"confidence >= {value:.2f}"
        points.append(score_cutoff(label, judgments, checked, choose, categories))
    return points


def policy_point(
    judgments: Mapping[str, Mapping[str, Any]],
    checked: Mapping[str, Sequence[Checked]],
    policy: GatePolicy,
    categories: Collection[str],
) -> CurvePoint:
    """Estimate the gate's own scores, to set against its official score.

    Args:
        judgments (Mapping[str, Mapping[str, Any]]): The ungated judgments, by PR.
        checked (Mapping[str, Sequence[Checked]]): The checked candidates, by PR.
        policy (GatePolicy): The gate's settings.
        categories (Collection[str]): Every SWR-Bench category code.

    Returns:
        CurvePoint: The estimate.
    """
    return score_cutoff(
        "gate (estimate)", judgments, checked, lambda found: gate(found, policy), categories
    )


def ungated_point(
    judgments: Mapping[str, Mapping[str, Any]],
    checked: Mapping[str, Sequence[Checked]],
    categories: Collection[str],
) -> CurvePoint:
    """Score every candidate with no gate, which must equal the ungated official score.

    Args:
        judgments (Mapping[str, Mapping[str, Any]]): The ungated judgments, by PR.
        checked (Mapping[str, Sequence[Checked]]): The checked candidates, by PR.
        categories (Collection[str]): Every SWR-Bench category code.

    Returns:
        CurvePoint: The scores of the whole ungated review.
    """
    return score_cutoff("ungated", judgments, checked, list, categories)
