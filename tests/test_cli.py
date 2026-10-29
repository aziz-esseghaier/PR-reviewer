"""Tests for the `pr-reviewer` command line in cli.py.

Checks that each command prints the right thing and that setup problems give a short hint with
exit code 1. No test calls a real LLM provider. The test marked `integration` uses a throwaway
database and is skipped when Postgres isn't running.

How to run (from the project folder):
    pytest tests/test_cli.py
"""

from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from pr_reviewer import __version__
from pr_reviewer.cli import main
from pr_reviewer.db.migrate import migrate

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


def test_demo_unknown_profile_lists_known_ones(
    models_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that `demo` with a typo in the profile exits 1 before touching the database."""
    assert main(["demo", "fake-smal", "--config", str(models_config)]) == 1
    assert "fake-small" in capsys.readouterr().err


@pytest.mark.integration
def test_db_migrate_creates_the_checkpoint_tables(
    test_db_url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that `db migrate` also sets up LangGraph's tables, and `db status` sees them."""
    monkeypatch.setenv("POSTGRES_DB", str(conninfo_to_dict(test_db_url)["dbname"]))
    assert main(["db", "migrate"]) == 0
    assert "checkpoint tables are ready" in capsys.readouterr().out
    assert main(["db", "status"]) == 0
    assert "Checkpoints: ready" in capsys.readouterr().out


@pytest.mark.integration
def test_cache_stats_reports_an_empty_cache(
    test_db_url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that `cache stats` works on a freshly migrated database and says it's empty."""
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        migrate(conn)
    # Point the command at the throwaway database instead of the real one.
    monkeypatch.setenv("POSTGRES_DB", str(conninfo_to_dict(test_db_url)["dbname"]))
    assert main(["cache", "stats"]) == 0
    assert "The cache is empty." in capsys.readouterr().out


@pytest.mark.integration
def test_usage_report_on_an_empty_log(
    test_db_url: str,
    models_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Check that `usage` lists each model's quota and says when nothing was sent yet."""
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        migrate(conn)
    monkeypatch.setenv("POSTGRES_DB", str(conninfo_to_dict(test_db_url)["dbname"]))
    assert main(["usage", "--config", str(models_config)]) == 0
    out = capsys.readouterr().out
    assert "fake fake-model-v1" in out
    assert "No requests yet." in out
