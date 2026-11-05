"""Small builders shared by the differential runner tests: PRs and finished runs.

Imported by the test modules; the file isn't run directly.
"""

import json

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.difftest.outcomes import DiffOutcome, Outcome, SideResult, Verdict
from pr_reviewer.difftest.runner import DiffResult
from tests.benchmark.test_dataset import make_row

# The verdicts at the two sides that give each outcome.
_SIDES: dict[Outcome, tuple[Verdict, Verdict]] = {
    "pass_fail": ("pass", "fail"),
    "pass_pass": ("pass", "pass"),
    "fail_pass": ("fail", "pass"),
    "fail_fail": ("fail", "fail"),
    "flaky": ("pass", "flaky"),
    "not_run": ("not_run", "pass"),
}


def make_pr(number: int = 1, issues: tuple[str, ...] = ()) -> Instance:
    """Make a 2022 psf/requests PR.

    Args:
        number (int): Its number.
        issues (tuple[str, ...]): Its known issues' change types, such as "F.2 Logic".

    Returns:
        Instance: The PR.
    """
    return Instance.model_validate_json(json.dumps(make_row(number, year=2022, issues=issues)))


def make_result(
    outcomes: dict[str, Outcome], old: str = "base", new: str = "pr", key: str = "k"
) -> DiffResult:
    """Make a finished differential run whose tests have the given outcomes.

    Args:
        outcomes (dict[str, Outcome]): Each test's outcome, by ID.
        old (str): The old side's name.
        new (str): The new side's name.
        key (str): Its request key.

    Returns:
        DiffResult: The run.
    """
    tests = []
    for test, outcome in outcomes.items():
        before, after = _SIDES[outcome]
        tests.append(
            DiffOutcome(
                test=test,
                outcome=outcome,
                old=SideResult(verdict=before, statuses=("passed",) * 3),
                new=SideResult(verdict=after, statuses=("failed",) * 3),
            )
        )
    return DiffResult(
        instance_id="psf__requests-1",
        old_side=old,
        new_side=new,
        targets=tuple(outcomes),
        tests=tuple(tests),
        problems={},
        minutes=1.5,
        key=key,
    )
