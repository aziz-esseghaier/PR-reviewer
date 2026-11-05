"""
Tests for judging validation cases and writing the validation results
(difftest/validation.py).

No Docker needed.

How to run (from the project folder):
    pytest tests/difftest/test_validation.py
"""

import json
from pathlib import Path

from pr_reviewer.difftest import validation
from pr_reviewer.difftest.outcomes import DiffOutcome, SideResult
from pr_reviewer.difftest.runner import DiffResult, FileSpec
from tests.difftest.helpers import make_result

FILES = (FileSpec("tests/test_a.py", "def test_a(): pass\n"),)
CONFIG = {"runs_per_side": 3, "difftest_minutes": 5}


def case(kind: validation.Kind) -> validation.Case:
    """Make a case of one kind.

    Args:
        kind (validation.Kind): Its kind.

    Returns:
        validation.Case: The case.
    """
    fix = "f" * 40 if kind == "fix_tests" else None
    return validation.Case(kind, "psf__requests-1", FILES, ("tests/test_a.py",), fix, "F.2")


def test_a_prs_own_tests_must_not_break() -> None:
    """Check fail->pass and pass->pass are expected for a PR's own tests, pass->fail isn't."""
    run = make_result({"t::a": "fail_pass", "t::b": "pass_pass", "t::c": "pass_fail"})
    judged = validation.judge_case(case("pr_tests"), "psf/requests", run, None)
    assert (judged.expected, judged.unexpected) == (2, ["t::c: pass_fail"])
    assert (judged.caught, judged.provable) == (None, 1)


def test_a_bug_is_caught_when_its_fix_test_fails_then_passes() -> None:
    """Check a fix case: caught from the PR state -> fix run, provable from the base."""
    base_pr = make_result({"t::a": "pass_fail", "t::b": "pass_pass"})
    pr_fix = make_result({"t::a": "fail_pass", "t::b": "pass_pass"}, old="pr", new="fix")
    judged = validation.judge_case(case("fix_tests"), "psf/requests", base_pr, pr_fix)
    assert (judged.caught, judged.provable, judged.unexpected) == (True, 1, [])
    missed = validation.judge_case(
        case("fix_tests"),
        "psf/requests",
        base_pr,
        make_result({"t::a": "flaky"}, old="pr", new="fix"),
    )
    assert (missed.caught, missed.unexpected) == (False, ["t::a: flaky"])


def test_related_tests_must_stay_as_they_were() -> None:
    """Check related tests may pass or fail at both sides, but not be flaky."""
    run = make_result({"t::a": "pass_pass", "t::b": "fail_fail", "t::c": "flaky"})
    judged = validation.judge_case(case("related"), "psf/requests", run, None)
    assert (judged.expected, judged.unexpected) == (2, ["t::c: flaky"])


def with_skipped(run: DiffResult, test: str) -> DiffResult:
    """Add a test skipped at both sides to a run.

    Args:
        run (DiffResult): The run.
        test (str): The skipped test's ID.

    Returns:
        DiffResult: The run with the test added.
    """
    side = SideResult(verdict="not_run", statuses=("skipped",) * 3, reason="skipped")
    skipped = DiffOutcome(test=test, outcome="not_run", old=side, new=side)
    return run.model_copy(update={"tests": (*run.tests, skipped)})


def test_tests_skipped_at_both_sides_are_neither_expected_nor_not() -> None:
    """Check skipped tests are counted apart, in judging and in the summary table."""
    run = with_skipped(make_result({"t::a": "pass_pass", "t::b": "not_run"}), "t::c[pdf]")
    judged = validation.judge_case(case("related"), "psf/requests", run, None)
    assert (judged.expected, judged.unexpected, judged.skipped) == (1, ["t::b: not_run"], 1)
    summary = validation.summary_markdown([judged], {}, CONFIG)
    assert "| related | 1 | 3 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 1 |" in summary
    pr_fix = with_skipped(make_result({"t::a": "fail_pass"}, old="pr", new="fix"), "t::d")
    fixed = validation.judge_case(case("fix_tests"), "psf/requests", run, pr_fix)
    assert (fixed.caught, fixed.skipped, fixed.unexpected) == (True, 1, [])
    assert "fail→pass 1, skipped 1" in validation.summary_markdown([fixed], {}, CONFIG)


def test_the_results_folder_has_tables_cases_and_raw_runs(tmp_path: Path) -> None:
    """Check summary.md's tables and the other files."""
    results = [
        validation.judge_case(
            case("pr_tests"), "psf/requests", make_result({"t::a": "fail_pass"}), None
        ),
        validation.judge_case(
            case("fix_tests"),
            "psf/requests",
            make_result({"t::a": "pass_fail"}),
            make_result({"t::a": "fail_pass"}, old="pr", new="fix"),
        ),
    ]
    validation.write_validation(tmp_path, results, {"psf__requests-2": ["a.py: no test"]}, CONFIG)
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "- 2 cases on 1 dev PRs" in summary
    assert "| pr_tests | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |" in summary
    assert "Caught 1 of 1 known bugs; 1 provable as pass→fail from the base." in summary
    assert "Every test's outcome was as expected." in summary
    assert "- psf__requests-2: a.py: no test" in summary
    cases = json.loads((tmp_path / "cases.json").read_text(encoding="utf-8"))
    assert [c["kind"] for c in cases] == ["pr_tests", "fix_tests"]
    assert cases[1]["caught"] is True
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) == CONFIG
    raw = (tmp_path / "raw" / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw) == 2
