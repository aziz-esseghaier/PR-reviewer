"""Shared builders for the evidence tests: test failures, side results and outcomes.

Imported by the test files in this folder; the file isn't run directly.
"""

from typing import Any

from pr_reviewer.difftest.outcomes import DiffOutcome, Outcome, SideResult
from pr_reviewer.difftest.reports import Failure

# The generated test's path in the examples.
TEST = "pkg/tests/test_pr_reviewer_f1.py"


def failure(
    exc: str = "AttributeError",
    frames: tuple[str, ...] = (f"{TEST}:5", "pkg/cart.py:11"),
    phase: str = "call",
    message: str = "AttributeError: 'NoneType' object has no attribute 'items'",
    classes: tuple[str, ...] | None = None,
) -> Failure:
    """Build a failure.

    Args:
        exc (str): Its exception type.
        frames (tuple[str, ...]): Its traceback, outermost first.
        phase (str): When it happened.
        message (str): Its message.
        classes (tuple[str, ...] | None): Its class and parents; None for (exc, "Exception",
            "BaseException").

    Returns:
        Failure: The failure.
    """
    return Failure(
        exc_type=exc,
        message=message,
        where=frames[-1] if frames else "",
        phase=phase,
        classes=classes if classes is not None else (exc, "Exception", "BaseException"),
        frames=frames,
    )


def side(verdict: str, fail: Failure | None = None, **more: Any) -> SideResult:
    """Build one side's result.

    Args:
        verdict (str): "pass", "fail", "flaky" or "not_run".
        fail (Failure | None): Its failure.
        **more (Any): Other fields, such as reason or hits.

    Returns:
        SideResult: The side.
    """
    status = {"pass": "passed", "fail": "failed"}.get(verdict, "failed")
    return SideResult.model_validate(
        {"verdict": verdict, "statuses": (status,) * 3, "failure": fail, **more}
    )


def outcome(old: SideResult, new: SideResult, test: str = TEST) -> DiffOutcome:
    """Build a test's outcome from its two sides.

    Args:
        old (SideResult): The base side.
        new (SideResult): The PR side.
        test (str): The test file.

    Returns:
        DiffOutcome: The outcome.
    """
    pairs: dict[tuple[str, str], Outcome] = {
        ("pass", "fail"): "pass_fail",
        ("pass", "pass"): "pass_pass",
        ("fail", "pass"): "fail_pass",
        ("fail", "fail"): "fail_fail",
    }
    if "not_run" in (old.verdict, new.verdict):
        kind: Outcome = "not_run"
    elif "flaky" in (old.verdict, new.verdict):
        kind = "flaky"
    else:
        kind = pairs[(old.verdict, new.verdict)]
    return DiffOutcome(test=f"{test}::test_x", outcome=kind, old=old, new=new)
