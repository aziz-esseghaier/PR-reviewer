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


def test_db_status_explains_unreachable_database(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that a down database gives a short hint and exit code 1, not a traceback."""
    monkeypatch.setenv("POSTGRES_USER", "nobody")
    monkeypatch.setenv("POSTGRES_PASSWORD", "unused")
    monkeypatch.setenv("POSTGRES_DB", "nothing")
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PORT", "1")  # nothing listens on port 1
    assert main(["db", "status"]) == 1
    assert "docker compose up -d" in capsys.readouterr().err
