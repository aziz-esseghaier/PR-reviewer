"""Tests for the command-line entry point."""

import pytest

from pr_reviewer import __version__
from pr_reviewer.cli import main


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Check that --version prints the installed package version and exits cleanly."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert __version__ in capsys.readouterr().out
