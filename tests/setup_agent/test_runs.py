"""
Tests for reading test results (testrun.py) and naming install failures (install.py).

Uses made-up command output, so no Docker is needed.

How to run (from the project folder):
    pytest tests/setup_agent/test_runs.py
"""

import datetime as dt

import pytest

from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.sandbox.runner import RunResult
from pr_reviewer.setup_agent.install import environment, failure_category, uv_settings
from pr_reviewer.setup_agent.testrun import judge_run, parse_django, parse_pytest


def ran(
    output: str, exit_code: int = 1, *, timed_out: bool = False, oom: bool = False
) -> RunResult:
    """Make a finished command with some output.

    Args:
        output (str): What it printed.
        exit_code (int): Its exit code.
        timed_out (bool): Whether it ran out of time.
        oom (bool): Whether it ran out of memory.

    Returns:
        RunResult: The result.
    """
    return RunResult(("x",), exit_code, 3.0, timed_out, oom, output, None)


def test_pytest_summary_lines_are_counted() -> None:
    """Check passed, failed, errors and skipped come from pytest's last summary line."""
    output = "....F\n= 1 failed, 45 passed, 3 skipped, 2 xfailed, 1 error in 3.21s ="
    assert parse_pytest(output) == (45, 1, 1, 5)
    assert parse_pytest("12 passed in 0.5s") == (12, 0, 0, 0)
    assert parse_pytest("no summary here, 3 passed earlier") is None


def test_django_summary_is_counted() -> None:
    """Check Django's "Ran N tests" and its FAILED line."""
    output = "....\nRan 40 tests in 2.0s\n\nFAILED (failures=2, errors=1, skipped=5)"
    assert parse_django(output) == (32, 2, 1, 5)
    assert parse_django("Ran 3 tests in 0.1s\n\nOK") == (3, 0, 0, 0)
    assert parse_django("nothing") is None


@pytest.mark.parametrize(
    ("output", "kwargs", "problem"),
    [
        ("95 passed, 5 failed in 2s", {}, None),
        ("89 passed, 11 failed in 2s", {}, "tests_failing"),
        (
            "2 errors in 1s\nE   ModuleNotFoundError: No module named 'hypothesis'",
            {},
            "test_import_errors",
        ),
        ("no tests ran in 0.1s", {}, "test_command_error"),
        ("5 deselected in 0.1s", {}, "no_tests_ran"),
        ("10 passed in 2s", {"timed_out": True}, "tests_timeout"),
        ("Killed", {"oom": True}, "tests_out_of_memory"),
    ],
)
def test_runs_pass_only_at_ninety_percent(
    output: str, kwargs: dict[str, bool], problem: str | None
) -> None:
    """Check the verdict: 90% must pass, and every other way a run can end is named."""
    outcome = judge_run(ran(output, **kwargs), "pytest")
    assert outcome.problem == problem
    assert outcome.ok is (problem is None)


@pytest.mark.parametrize(
    ("output", "category"),
    [
        ("ERROR: Tunnel connection failed: 403 Filtered", "network_blocked"),
        ("error: No download found for request: cpython-3.4", "python_unavailable"),
        ("x  No solution found when resolving dependencies:", "dependency_resolution"),
        ("src/a.c:1:10: fatal error: zlib.h: No such file or directory", "missing_system_library"),
        ("error: command '/usr/bin/gcc' failed with exit code 1", "build_failed"),
        ("something else went wrong", "install_failed"),
    ],
)
def test_install_failures_are_named_from_the_log(output: str, category: str) -> None:
    """Check each failure cause is found in the output, and an unknown one gets the fallback."""
    assert failure_category(ran(output)) == category


def test_install_timeouts_and_memory_are_named_first() -> None:
    """Check running out of time or memory wins over whatever the output says."""
    assert failure_category(ran("Failed to build", timed_out=True)) == "install_timeout"
    assert failure_category(ran("Failed to build", oom=True)) == "install_out_of_memory"


def test_environment_pins_packages_to_the_pr_date() -> None:
    """Check the variables: the venv first, UV_EXCLUDE_NEWER at the PR date, and a recipe can't
    change ours."""
    recipe = EnvRecipe(
        repo="psf/requests",
        python_version="3.9",
        install_steps=("uv pip install -e .",),
        test_command="python -m pytest",
        env={"MPLBACKEND": "agg", "UV_EXCLUDE_NEWER": "2030-01-01T00:00:00Z", "PATH": "/evil"},
    )
    env = environment(recipe, dt.datetime(2022, 3, 1, 12, 30, tzinfo=dt.UTC))
    assert env["UV_EXCLUDE_NEWER"] == "2022-03-01T12:30:00Z"
    assert env["PATH"].startswith("/work/venv/bin:")
    assert env["MPLBACKEND"] == "agg"
    assert env["UV_CONFIG_FILE"] == "/tmp/pr_reviewer_uv.toml"


def test_old_prs_may_use_setuptools_that_does_editable_installs() -> None:
    """Check only PRs before September 2022 get the setuptools exception."""
    old = uv_settings(dt.datetime(2020, 5, 1, tzinfo=dt.UTC))
    assert old == 'exclude-newer-package = { setuptools = "2022-09-01T00:00:00Z" }\n'
    assert uv_settings(dt.datetime(2023, 1, 1, tzinfo=dt.UTC)) == ""
