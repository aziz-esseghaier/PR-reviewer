"""
Tests for the `pr-reviewer sandbox ...` commands in sandbox/commands.py.

Checks the refusals that come before any container starts: `sandbox run` keeps the test set
locked, and needs a command. Uses the small dataset and split from tests/test_cli.py. No
Docker needed.

How to run (from the project folder):
    pytest tests/sandbox/test_commands.py
"""

from pathlib import Path

import pytest

from pr_reviewer.benchmark.source import sha256_of
from pr_reviewer.cli import build_parser, main
from tests.test_cli import tiny_dataset  # noqa: F401  (a fixture, found by pytest by name)


@pytest.fixture
def sandbox_dataset(tiny_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> Path:  # noqa: F811
    """Point the sandbox commands at the small dataset and split too.

    Args:
        tiny_dataset (Path): The dataset file; its split is saved next to it.
        monkeypatch (pytest.MonkeyPatch): Used to swap the pinned files for this test only.

    Returns:
        Path: The dataset file.
    """
    split_file = tiny_dataset.parent / "split.json"
    pinned = PinnedLike(tiny_dataset)
    monkeypatch.setattr("pr_reviewer.sandbox.commands.DATASET", pinned)
    monkeypatch.setattr("pr_reviewer.sandbox.commands.split_path", lambda: split_file)
    monkeypatch.setattr("pr_reviewer.sandbox.commands.SPLIT_SHA256", sha256_of(split_file))
    return tiny_dataset


class PinnedLike:
    """Stands in for the pinned dataset entry: only its fingerprint is read.

    Args:
        path (Path): The dataset file whose fingerprint to give.
    """

    def __init__(self, path: Path) -> None:
        self.sha256 = sha256_of(path)


def test_run_keeps_the_test_set_locked(
    sandbox_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check `sandbox run` refuses a test-set PR before starting anything."""
    code = main(["sandbox", "run", "psf__requests-2", "--data", str(sandbox_dataset), "--", "ls"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Refused: psf__requests-2 is in the test set" in captured.err
    assert captured.out == ""


def test_run_needs_a_command(sandbox_dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Check `sandbox run` without a command says how to give one."""
    assert main(["sandbox", "run", "psf__requests-5", "--data", str(sandbox_dataset)]) == 1
    assert "after --, such as: -- git log -1" in capsys.readouterr().err


def test_run_reads_options_before_the_command() -> None:
    """Check options between the PR and "--" are options, and the rest is the command."""
    args = build_parser().parse_args(
        ["sandbox", "run", "x__y-1", "--at", "base", "--network", "--", "git", "log", "-1"]
    )
    assert (args.instance_id, args.at, args.network) == ("x__y-1", "base", True)
    assert args.command == ["git", "log", "-1"]
