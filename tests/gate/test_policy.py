"""Tests for the gate's rules (gate/policy.py).

Checks the tier and confidence filters, one comment per place, the cap, the best-first order,
and reading configs/gate.yaml.

How to run (from the project folder):
    pytest tests/gate/test_policy.py
"""

from pathlib import Path

import pytest

from pr_reviewer.core.config import ConfigError
from pr_reviewer.core.schemas import Claim
from pr_reviewer.gate.policy import Checked, GatePolicy, gate, load_policy


def checked(
    fid: str, tier: int, confidence: float, lines: tuple[int, int] = (10, 12), order: int = 0
) -> Checked:
    """Build a checked candidate on pkg/cart.py.

    Args:
        fid (str): The finding ID.
        tier (int): Its tier.
        confidence (float): The reviewer's confidence.
        lines (tuple[int, int]): Its first and last line.
        order (int): Its place in the reviewer's list.

    Returns:
        Checked: The candidate.
    """
    claim = Claim(
        file="pkg/cart.py",
        start_line=lines[0],
        end_line=lines[1],
        category="F.4",
        summary=f"{fid}: total breaks on None.",
        confidence=confidence,
    )
    return Checked("pkg__pkg-1", fid, order, claim, tier)


def policy(**changes: object) -> GatePolicy:
    """Build the default gate settings, with some changed.

    Args:
        **changes (object): Settings to replace.

    Returns:
        GatePolicy: The settings.
    """
    settings = {"max_tier": 1, "min_confidence": 0.0, "dedupe": True, "max_per_pr": 3}
    return GatePolicy.model_validate({**settings, **changes})


def test_only_the_tiers_and_confidence_allowed_pass() -> None:
    """Check tier 2 is dropped under tier 1 only, and low confidence under a threshold."""
    found = [
        checked("f1", 1, 0.5, (1, 1), 0),
        checked("f2", 2, 0.9, (5, 5), 1),
        checked("f3", 1, 0.9, (9, 9), 2),
    ]
    assert [c.finding_id for c in gate(found, policy())] == ["f3", "f1"]
    assert [c.finding_id for c in gate(found, policy(min_confidence=0.8))] == ["f3"]
    assert [c.finding_id for c in gate(found, policy(max_tier=2))] == ["f3", "f1", "f2"]


def test_one_comment_per_place_keeps_the_best() -> None:
    """Check overlapping findings keep the best tier, then the highest confidence."""
    found = [
        checked("f1", 2, 0.9, (10, 14), 0),
        checked("f2", 1, 0.6, (12, 20), 1),
        checked("f3", 1, 0.8, (19, 19), 2),
        checked("f4", 1, 0.7, (30, 31), 3),
    ]
    kept = gate(found, policy(max_tier=4, max_per_pr=None))
    # f3 (tier 1, 0.8) first; f2 overlaps it; f4 is elsewhere; f1 doesn't overlap f3.
    assert [c.finding_id for c in kept] == ["f3", "f4", "f1"]
    kept = gate(found, policy(max_tier=4, max_per_pr=None, dedupe=False))
    assert [c.finding_id for c in kept] == ["f3", "f4", "f2", "f1"]


def test_the_cap_keeps_the_best_few() -> None:
    """Check at most max_per_pr comments are kept, best first."""
    found = [checked(f"f{n}", 1, 0.5 + n / 10, (n * 10 + 1, n * 10 + 1), n) for n in range(5)]
    kept = gate(found, policy(max_per_pr=2))
    assert [c.finding_id for c in kept] == ["f4", "f3"]


def test_the_settings_file_is_read_and_checked(tmp_path: Path) -> None:
    """Check the project's gate.yaml loads, and a bad file is refused."""
    assert load_policy() == policy()
    bad = tmp_path / "gate.yaml"
    bad.write_text("max_tier: 5\nmin_confidence: 0\ndedupe: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="isn't a valid gate config"):
        load_policy(bad)
