"""Tests for the command-line entry point."""

from pathlib import Path

import pytest

from pr_reviewer import __version__
from pr_reviewer.cli import main

MODELS_CONFIG = """
providers:
  fake:
    base_url: https://llm.example.test/v1
    api_key_env: FAKE_LLM_KEY
models:
  fake-small:
    provider: fake
    model: fake-model-v1
"""


@pytest.fixture
def models_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a model config with one fake provider whose key is not set.

    Args:
        tmp_path (Path): Temporary folder for this test.
        monkeypatch (pytest.MonkeyPatch): Used to unset the fake key for this test only.

    Returns:
        Path: The config file.
    """
    monkeypatch.delenv("FAKE_LLM_KEY", raising=False)
    path = tmp_path / "models.yaml"
    path.write_text(MODELS_CONFIG, encoding="utf-8")
    return path


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


def test_llm_models_lists_profiles_and_key_state(
    models_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that `llm models` lists each profile and says when its key is missing."""
    assert main(["llm", "models", "--config", str(models_config)]) == 0
    out = capsys.readouterr().out
    assert "fake-small" in out
    assert "FAKE_LLM_KEY missing" in out


def test_llm_ping_without_key_explains_the_problem(
    models_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that pinging a profile whose key is missing exits 1 and names the variable."""
    assert main(["llm", "ping", "fake-small", "--config", str(models_config)]) == 1
    assert "FAKE_LLM_KEY is not set" in capsys.readouterr().err


def test_llm_ping_unknown_profile_lists_known_ones(
    models_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that a typo in the profile name exits 1 and shows the valid names."""
    assert main(["llm", "ping", "fake-smal", "--config", str(models_config)]) == 1
    assert "fake-small" in capsys.readouterr().err
