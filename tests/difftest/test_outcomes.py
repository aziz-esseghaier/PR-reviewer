"""
Tests for turning runs at two sides into one outcome per test (difftest/outcomes.py).

No Docker needed.

How to run (from the project folder):
    pytest tests/difftest/test_outcomes.py
"""

from pr_reviewer.difftest import outcomes
from pr_reviewer.difftest.reports import Failure, Result, RunReport, Status

BOOM = Failure(exc_type="AttributeError", message="no attribute", where="a.py:3", phase="call")
GONE = Failure(exc_type="ImportError", message="cannot import", where="", phase="collect")


def run(
    statuses: dict[str, Status] | None = None,
    collect: dict[str, Failure] | None = None,
    problem: str | None = None,
) -> RunReport:
    """Make one run's report.

    Args:
        statuses (dict[str, Status] | None): Each test's status.
        collect (dict[str, Failure] | None): Files that failed to load.
        problem (str | None): Why the run couldn't run.

    Returns:
        RunReport: The report.
    """
    results = {
        test: Result(test=test, status=status, failure=BOOM if status != "passed" else None)
        for test, status in (statuses or {}).items()
    }
    return RunReport(
        results=results, collect_errors=collect or {}, problem=problem, detail="", seconds=1.0
    )


def test_a_side_is_pass_fail_or_flaky_by_its_runs() -> None:
    """Check the verdict at one side from three runs."""
    test = "t.py::test_a"
    passing = [run({test: "passed"})] * 3
    failing = [run({test: "failed"}), run({test: "error"}), run({test: "failed"})]
    mixed = [run({test: "passed"}), run({test: "failed"}), run({test: "passed"})]
    assert outcomes.side_result(test, passing, "pytest").verdict == "pass"
    failed = outcomes.side_result(test, failing, "pytest")
    assert (failed.verdict, failed.failure) == ("fail", BOOM)
    flaky = outcomes.side_result(test, mixed, "pytest")
    assert (flaky.verdict, flaky.statuses) == ("flaky", ("passed", "failed", "passed"))


def test_timeouts_skips_and_missing_tests_are_not_run() -> None:
    """Check a run that couldn't run, a skipped test and an uncollected one are not_run."""
    test = "t.py::test_a"
    timed_out = outcomes.side_result(
        test, [run({test: "passed"}), run(problem="timeout"), run({test: "passed"})], "pytest"
    )
    assert (timed_out.verdict, timed_out.reason) == ("not_run", "timeout")
    skipped = outcomes.side_result(test, [run({test: "skipped"})] * 3, "pytest")
    assert (skipped.verdict, skipped.reason) == ("not_run", "skipped")
    missing = outcomes.side_result(test, [run({})] * 3, "pytest")
    assert (missing.verdict, missing.reason) == ("not_run", "not collected")


def test_a_file_that_fails_to_load_fails_its_tests() -> None:
    """Check tests of a file with a collection error count as errors with that failure."""
    found = outcomes.side_result("t.py::test_a", [run(collect={"t.py": GONE})] * 3, "pytest")
    assert (found.verdict, found.statuses, found.failure) == ("fail", ("error",) * 3, GONE)


def test_a_file_skipped_while_loading_skips_its_tests() -> None:
    """Check tests of a file skipped at import (importorskip) are skipped, not uncollected."""
    skipped = run().model_copy(update={"collect_skips": {"t.py": "could not import 'sparse'"}})
    found = outcomes.side_result("t.py::test_a", [skipped] * 3, "pytest")
    assert (found.verdict, found.reason, found.statuses) == ("not_run", "skipped", ("skipped",) * 3)


def test_a_run_that_stopped_early_counts_for_none_of_its_results() -> None:
    """Check a run pytest stopped (such as for a test ID it can't find) is not_run."""
    stopped = run({"t.py::test_a": "passed"}, problem="usage_error")
    found = outcomes.side_result("t.py::test_a", [stopped] * 3, "pytest")
    assert (found.verdict, found.reason) == ("not_run", "usage_error")


def test_outcomes_put_not_run_before_flaky_before_the_four_pairs() -> None:
    """Check how the two sides' verdicts give the outcome."""
    side = {
        verdict: outcomes.SideResult(verdict=verdict, statuses=())
        for verdict in ("pass", "fail", "flaky", "not_run")
    }
    assert outcomes.combine(side["pass"], side["fail"]) == "pass_fail"
    assert outcomes.combine(side["pass"], side["pass"]) == "pass_pass"
    assert outcomes.combine(side["fail"], side["pass"]) == "fail_pass"
    assert outcomes.combine(side["fail"], side["fail"]) == "fail_fail"
    assert outcomes.combine(side["pass"], side["flaky"]) == "flaky"
    assert outcomes.combine(side["flaky"], side["not_run"]) == "not_run"


def test_compare_expands_targets_and_keeps_targets_nothing_reported() -> None:
    """Check a file target expands to its tests, and a target with no result stays listed."""
    old = [run({"t.py::test_a": "passed", "t.py::test_b[1]": "passed"})] * 3
    new = [run({"t.py::test_a": "failed", "t.py::test_b[1]": "passed"})] * 3
    found = outcomes.compare(["t.py::test_b", "t.py", "u.py::test_c"], old, new, "pytest")
    assert [(f.test, f.outcome) for f in found] == [
        ("t.py::test_b[1]", "pass_pass"),
        ("t.py::test_a", "pass_fail"),
        ("u.py::test_c", "not_run"),
    ]


def test_targets_cover_tests_inside_them() -> None:
    """Check which test IDs fall under a target, for pytest and Django."""
    assert outcomes.covers("t.py", "t.py::T::test_a", "pytest")
    assert outcomes.covers("t.py::test_a", "t.py::test_a[2]", "pytest")
    assert not outcomes.covers("t.py::test_a", "t.py::test_ab", "pytest")
    assert outcomes.covers("app.tests", "app.tests.T.test_a", "django")
    assert not outcomes.covers("app.tests", "app.tests_more.T.test_a", "django")
