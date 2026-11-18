"""The gate's rules: which checked findings of a PR become review comments.

The evidence agent gives each candidate a tier (1 = executed proof, 4 = no evidence). The gate
keeps a candidate when:
- its tier is at most max_tier, and its confidence is at least min_confidence;
- no better finding already kept covers the same place (same file, overlapping lines), so one
  problem never gets two comments (dedupe);
- the PR has fewer than max_per_pr comments so far, so a noisy PR can't flood the review.
Findings are taken best first: lower tier, then higher confidence, then the reviewer's order.
Taking them in that order means a looser cutoff only adds comments, never swaps one out, which
keeps the precision/recall curve (curve.py) well defined.

The settings live in configs/gate.yaml, chosen on the dev set and frozen before the test run.

Imported by other modules; the file isn't run directly.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pr_reviewer.core.config import ConfigError, project_root
from pr_reviewer.core.schemas import Claim

# The gate's settings file, under the project folder.
GATE_CONFIG = Path("configs") / "gate.yaml"


class GatePolicy(BaseModel):
    """The gate's settings.

    Attributes:
        max_tier (int): Weakest tier kept, 1 to 4.
        min_confidence (float): Lowest confidence kept, 0 to 1.
        dedupe (bool): Keep one finding per place (same file, overlapping lines).
        max_per_pr (int | None): Most comments per PR; None for no cap.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tier: int = Field(ge=1, le=4)
    min_confidence: float = Field(ge=0.0, le=1.0)
    dedupe: bool
    max_per_pr: int | None = Field(default=None, ge=1)


def load_policy(path: Path | None = None) -> GatePolicy:
    """Read the gate's settings.

    Args:
        path (Path | None): The YAML file; None means configs/gate.yaml in the project.

    Returns:
        GatePolicy: The settings.

    Raises:
        ConfigError: If the file is missing or its settings aren't valid.
    """
    file = path or project_root() / GATE_CONFIG
    if not file.is_file():
        raise ConfigError(f"{file} not found.")
    try:
        return GatePolicy.model_validate(yaml.safe_load(file.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"{file} isn't a valid gate config: {exc}") from None


@dataclass(frozen=True)
class Checked:
    """A candidate with the tier the evidence agent gave it.

    Attributes:
        instance_id (str): The PR.
        finding_id (str): The candidate, such as "f3".
        order (int): Its place in the reviewer's list, from 0.
        claim (Claim): What the reviewer said.
        tier (int): Its evidence tier, 1 to 4.
    """

    instance_id: str
    finding_id: str
    order: int
    claim: Claim
    tier: int

    @property
    def confidence(self) -> float:
        """The reviewer's confidence in the claim.

        Returns:
            float: 0 to 1.
        """
        return self.claim.confidence


def rank(item: Checked) -> tuple[int, float, int]:
    """Give the gate's sort key: lower tier, then higher confidence, then reviewer order.

    Args:
        item (Checked): The finding.

    Returns:
        tuple[int, float, int]: The key; smaller comes first.
    """
    return (item.tier, -item.confidence, item.order)


def overlaps(a: Checked, b: Checked) -> bool:
    """Tell whether two findings point at the same place: one file, overlapping lines.

    Args:
        a (Checked): One finding.
        b (Checked): The other.

    Returns:
        bool: True when they share a file and at least one line.
    """
    return (
        a.claim.file == b.claim.file
        and a.claim.start_line <= b.claim.end_line
        and b.claim.start_line <= a.claim.end_line
    )


def take(
    found: Sequence[Checked],
    keep: Callable[[Checked], bool],
    key: Callable[[Checked], Any],
    dedupe: bool,
    cap: int | None,
) -> list[Checked]:
    """Take one PR's findings best first, skipping repeats of a place, up to a cap.

    Args:
        found (Sequence[Checked]): The PR's findings.
        keep (Callable[[Checked], bool]): Which ones may be taken, such as tier 1 only.
        key (Callable[[Checked], Any]): The sort key; smaller is better.
        dedupe (bool): Skip a finding that overlaps one already taken.
        cap (int | None): Most findings taken; None for no cap.

    Returns:
        list[Checked]: The findings taken, best first.
    """
    taken: list[Checked] = []
    for item in sorted((item for item in found if keep(item)), key=key):
        if cap is not None and len(taken) >= cap:
            break
        if dedupe and any(overlaps(item, other) for other in taken):
            continue
        taken.append(item)
    return taken


def gate(found: Sequence[Checked], policy: GatePolicy) -> list[Checked]:
    """Pick the findings of one PR that become comments.

    Args:
        found (Sequence[Checked]): The PR's checked findings.
        policy (GatePolicy): The gate's settings.

    Returns:
        list[Checked]: The comments, best first.
    """

    def passes(item: Checked) -> bool:
        return item.tier <= policy.max_tier and item.confidence >= policy.min_confidence

    return take(found, passes, rank, policy.dedupe, policy.max_per_pr)
