"""Turning the runs at two sides of a PR into one outcome per test.

Each side's tests run several times (RUNS_PER_SIDE in runner.py). Per side, a test is:
- pass: it passed in every run;
- fail: it failed or errored in every run;
- flaky: the runs disagree, so its result is chance;
- not_run: it never got a fair chance: a run timed out, ran out of memory, the runner broke
  or stopped early (such as pytest refusing a test ID), or the test was skipped or never
  collected.
Then the two sides, old (the base) and new (the PR state), give the outcome:
- "pass_fail": passes before the PR, fails after it. The PR broke what the test checks: the
  one outcome that can prove a bug;
- "pass_pass", "fail_pass", "fail_fail": the other three;
- "flaky": flaky at either side, so it can never count as proof;
- "not_run": not run at either side, such as after a timeout.
not_run wins over flaky, so a timeout is never mistaken for proof.

Imported by other modules; the file isn't run directly.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from pr_reviewer.difftest.reports import Failure, RunReport

# A test's outcome across the two sides.
Outcome = Literal["pass_fail", "pass_pass", "fail_pass", "fail_fail", "flaky", "not_run"]
OUTCOMES: tuple[Outcome, ...] = (
    "pass_fail",
    "pass_pass",
    "fail_pass",
    "fail_fail",
    "flaky",
    "not_run",
)

# How each outcome is written in tables and on screen.
LABELS: dict[Outcome, str] = {
    "pass_fail": "pass→fail",
    "pass_pass": "pass→pass",
    "fail_pass": "fail→pass",
    "fail_fail": "fail→fail",
    "flaky": "flaky",
    "not_run": "not run",
}

# A test's verdict at one side.
Verdict = Literal["pass", "fail", "flaky", "not_run"]

# The outcome of each pair of steady verdicts, old side first.
_PAIRS: dict[tuple[Verdict, Verdict], Outcome] = {
    ("pass", "fail"): "pass_fail",
    ("pass", "pass"): "pass_pass",
    ("fail", "pass"): "fail_pass",
    ("fail", "fail"): "fail_fail",
}

# The test runner's kind: pytest node IDs or Django labels.
Runner = Literal["pytest", "django"]


class SideResult(BaseModel):
    """One test at one side, over all its runs.

    Attributes:
        verdict (Verdict): "pass", "fail", "flaky" or "not_run".
        statuses (tuple[str, ...]): What each run gave: "passed", "failed", "error",
            "skipped", "missing" (not collected), or the run's problem, such as "timeout".
        failure (Failure | None): The first failure seen, if any.
        reason (str): Why it's not_run, such as "timeout" or "skipped"; "" otherwise.
        hits (tuple[str, ...]): The watched lines it ran in any run, such as "pkg/a.py:12",
            sorted; empty when no lines were watched.
    """

    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    statuses: tuple[str, ...]
    failure: Failure | None = None
    reason: str = ""
    hits: tuple[str, ...] = ()


class DiffOutcome(BaseModel):
    """One test's outcome from the old side to the new one.

    Attributes:
        test (str): The test's ID.
        outcome (Outcome): Such as "pass_fail".
        old (SideResult): How it did at the old side (the base).
        new (SideResult): How it did at the new side (the PR state).
    """

    model_config = ConfigDict(frozen=True)

    test: str
    outcome: Outcome
    old: SideResult
    new: SideResult


def covers(target: str, test: str, runner: Runner) -> bool:
    """Tell whether a test ID falls under a requested target.

    Args:
        target (str): What was asked for: a file, a class or a test, such as
            "tests/test_a.py", "tests/test_a.py::T" or "app.tests.T".
        test (str): A test's ID, such as "tests/test_a.py::T::test_b[1]".
        runner (Runner): "pytest" or "django".

    Returns:
        bool: True when the test is the target or inside it.
    """
    if test == target:
        return True
    if runner == "django":
        return test.startswith(f"{target}.")
    return test.startswith((f"{target}::", f"{target}["))


def _status_in(test: str, run: RunReport, runner: Runner) -> tuple[str, Failure | None]:
    """Give a test's status in one run.

    Args:
        test (str): The test's ID.
        run (RunReport): The run.
        runner (Runner): "pytest" or "django".

    Returns:
        tuple[str, Failure | None]: The status, with its failure when it failed. A test whose
        file or class failed to load is an "error" with that failure; one whose file or class
        was skipped while loading is "skipped".
    """
    if run.problem:
        return run.problem, None
    result = run.results.get(test)
    if result is not None:
        return result.status, result.failure
    for owner, failure in run.collect_errors.items():
        if covers(owner, test, runner):
            return "error", failure
    if any(covers(owner, test, runner) for owner in run.collect_skips):
        return "skipped", None
    return "missing", None


def side_result(test: str, runs: Sequence[RunReport], runner: Runner) -> SideResult:
    """Give one test's verdict at one side from all its runs there.

    Args:
        test (str): The test's ID.
        runs (Sequence[RunReport]): The side's runs.
        runner (Runner): "pytest" or "django".

    Returns:
        SideResult: The verdict, each run's status and the first failure.
    """
    seen = [_status_in(test, run, runner) for run in runs]
    statuses = tuple(status for status, _ in seen)
    failure = next((failure for _, failure in seen if failure is not None), None)
    hits = tuple(
        sorted({hit for run in runs if test in run.results for hit in run.results[test].hits})
    )
    known = {"passed", "failed", "error", "skipped", "missing"}
    problems = [status for status in statuses if status not in known]
    if problems or not statuses:
        reason = problems[0] if problems else "no runs"
        return SideResult(
            verdict="not_run", statuses=statuses, failure=failure, reason=reason, hits=hits
        )
    if all(status == "missing" for status in statuses):
        return SideResult(verdict="not_run", statuses=statuses, reason="not collected")
    if all(status == "skipped" for status in statuses):
        return SideResult(verdict="not_run", statuses=statuses, reason="skipped", hits=hits)
    if all(status == "passed" for status in statuses):
        return SideResult(verdict="pass", statuses=statuses, hits=hits)
    if all(status in ("failed", "error") for status in statuses):
        return SideResult(verdict="fail", statuses=statuses, failure=failure, hits=hits)
    return SideResult(verdict="flaky", statuses=statuses, failure=failure, hits=hits)


def combine(old: SideResult, new: SideResult) -> Outcome:
    """Give a test's outcome from its verdicts at the two sides.

    Args:
        old (SideResult): Its verdict at the old side (the base).
        new (SideResult): Its verdict at the new side (the PR state).

    Returns:
        Outcome: "not_run" if either side didn't run, then "flaky" if either is flaky,
        otherwise one of the four pass/fail pairs.
    """
    if "not_run" in (old.verdict, new.verdict):
        return "not_run"
    if "flaky" in (old.verdict, new.verdict):
        return "flaky"
    return _PAIRS[old.verdict, new.verdict]


def compare(
    targets: Sequence[str],
    old_runs: Sequence[RunReport],
    new_runs: Sequence[RunReport],
    runner: Runner,
) -> list[DiffOutcome]:
    """Give every requested test's outcome from the runs at the two sides.

    The tests are those any run reported under the targets, so a target naming a file or a
    parametrized test expands to every test in it. A target no run reported (such as a file
    that never loaded) is kept as one test of its own, so nothing asked for goes missing.

    Args:
        targets (Sequence[str]): What was asked to run: files, classes or tests.
        old_runs (Sequence[RunReport]): The runs at the old side.
        new_runs (Sequence[RunReport]): The runs at the new side.
        runner (Runner): "pytest" or "django".

    Returns:
        list[DiffOutcome]: One outcome per test, in target order, then test ID order.
    """
    reported = {test for run in (*old_runs, *new_runs) for test in run.results}
    tests: list[str] = []
    for target in targets:
        under = sorted(test for test in reported if covers(target, test, runner))
        for test in under or [target]:
            if test not in tests:
                tests.append(test)
    outcomes = []
    for test in tests:
        old = side_result(test, old_runs, runner)
        new = side_result(test, new_runs, runner)
        outcomes.append(DiffOutcome(test=test, outcome=combine(old, new), old=old, new=new))
    return outcomes
