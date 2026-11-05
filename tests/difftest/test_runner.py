"""
Tests for the runner's pieces that need no sandbox (difftest/runner.py): test file paths,
request keys, outcome counts and the plugin's source.

The real runs, in sandboxes, are in test_difftest_docker.py. No Docker needed here.

How to run (from the project folder):
    pytest tests/difftest/test_runner.py
"""

from typing import cast

import pytest

from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.difftest import runner
from pr_reviewer.setup_agent.attempt import TryContext
from tests.difftest.helpers import make_pr, make_result

RECIPE = EnvRecipe(
    repo="psf/requests",
    python_version="3.9",
    install_steps=("uv pip install -e .",),
    test_command="python -m pytest",
)


def test_test_files_must_stay_inside_the_repo() -> None:
    """Check absolute paths, ".." and backslashes are refused."""
    assert runner.FileSpec("pkg/tests/test_a.py", "").path == "pkg/tests/test_a.py"
    for bad in ("", "/etc/passwd", "../outside.py", "pkg/../../x.py", "pkg\\test_a.py"):
        with pytest.raises(ValueError, match="inside the repo"):
            runner.FileSpec(bad, "")


def test_the_request_key_follows_what_decides_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the key changes with the images, test code and targets, and only with them."""
    monkeypatch.setattr(runner, "image_id", lambda ctx, image: f"sha256:{image}")
    ctx = cast(TryContext, None)
    pr = make_pr()
    sides = (runner.SideSpec("base", "img-base"), runner.SideSpec("pr", "img-pr"))
    files = (runner.FileSpec("tests/test_a.py", "def test_a(): pass\n"),)

    def key(**changes: object) -> str:
        args: dict[str, object] = {
            "sides": sides,
            "files": files,
            "targets": ("tests/test_a.py",),
            "runs": 3,
            "timeout": 300.0,
        }
        args.update(changes)
        return runner.request_key(ctx, pr, RECIPE, **args)  # type: ignore[arg-type]

    same = key()
    assert key() == same
    assert key(files=(runner.FileSpec("tests/test_a.py", "def test_a(): 1\n"),)) != same
    assert key(targets=("tests/test_a.py::test_a",)) != same
    assert key(sides=(sides[0], runner.SideSpec("pr", "img-other"))) != same
    assert key(sides=(sides[0], runner.SideSpec("fix", "img-pr", "a" * 40))) != same
    assert key(runs=1) != same


def test_counts_cover_every_outcome() -> None:
    """Check a run's counts list every outcome, with zeros."""
    counts = make_result({"t::a": "pass_fail", "t::b": "pass_fail", "t::c": "flaky"}).counts()
    assert counts == {
        "pass_fail": 2,
        "pass_pass": 0,
        "fail_pass": 0,
        "fail_fail": 0,
        "flaky": 1,
        "not_run": 0,
    }


def test_a_run_needs_targets_and_two_different_sides() -> None:
    """Check bad requests are refused before any sandbox starts."""
    ctx = cast(TryContext, None)
    base, pr_side = runner.SideSpec("base", "a"), runner.SideSpec("pr", "b")
    with pytest.raises(ValueError, match="at least one test"):
        runner.run_diff(ctx, make_pr(), RECIPE, base, pr_side, (), ())
    with pytest.raises(ValueError, match="different names"):
        runner.run_diff(ctx, make_pr(), RECIPE, base, base, (), ("t.py",))


def test_the_plugin_ships_with_the_package() -> None:
    """Check the plugin's source can be read, to copy into sandboxes."""
    source = runner.probe_source()
    assert "PR_REVIEWER_REPORT" in source
    assert "def pytest_runtest_logreport" in source
