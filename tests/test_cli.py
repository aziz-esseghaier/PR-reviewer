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
from pr_reviewer.benchmark.source import DATASET, PinnedFile, sha256_of
from pr_reviewer.benchmark.split import Split, save_split
from pr_reviewer.cli import main
from pr_reviewer.db.migrate import migrate
from tests.benchmark.test_dataset import make_row, sha, write_dataset

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


@pytest.fixture
def tiny_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a five-PR dataset and its split, and pin the bench commands to both.

    Three PRs are benchmark PRs: psf__requests-5 is in dev, psf__requests-2 and
    pallets__flask-4 are in test. psf__requests-1 is too old and psf__requests-3 has only
    evolvability issues.

    Args:
        tmp_path (Path): Temporary folder for this test.
        monkeypatch (pytest.MonkeyPatch): Used to swap the pinned files for this test only.

    Returns:
        Path: The dataset file.
    """
    rows = [
        make_row(1, year=2018),
        make_row(2, year=2019),
        make_row(3, year=2019, issues=("E.3.2 Solution Approach",)),
        make_row(4, year=2020, issues=(), repo="pallets/flask"),
        make_row(5, year=2021, issues=("E.2 Visual Representation", "F.4 Check")),
    ]
    path = write_dataset(tmp_path, rows)
    pinned = PinnedFile(DATASET.path, sha256_of(path), path.stat().st_size)
    monkeypatch.setattr("pr_reviewer.cli.DATASET", pinned)
    split = Split(
        seed=0,
        dev_fraction=0.3,
        source_commit="0" * 40,
        dataset_sha256=pinned.sha256,
        dev=("psf__requests-5",),
        test=("pallets__flask-4", "psf__requests-2"),
    )
    split_file = tmp_path / "split.json"
    save_split(split, split_file)
    monkeypatch.setattr("pr_reviewer.cli.split_path", lambda: split_file)
    monkeypatch.setattr("pr_reviewer.cli.SPLIT_SHA256", sha256_of(split_file))
    return path


def test_bench_info_counts_the_benchmark_prs(
    tiny_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check the summary and the per-repo and per-year tables of `bench info`."""
    assert main(["bench", "info", "--data", str(tiny_dataset)]) == 0
    out = capsys.readouterr().out
    assert "All PRs:       5 (4 with known issues)" in out
    assert "Benchmark PRs: 3 from 2019 onward" in out
    assert "2 with a functional issue (1 also have evolvability issues)" in out
    assert "Known issues in them: 2 functional, 1 evolvability" in out
    assert "Left out: 1 PRs from before 2019, and 1 from 2019 onward" in out
    assert "pallets/flask           0      1      1    0     1" in out
    assert "psf/requests            2      0      2    1     1" in out
    assert "2019              1      0      1    0     1" in out
    assert "unchanged copies in third_party/swrbench/" in out
    assert "0 inputs over 8,000 tokens" in out
    assert "Split: 1 dev PRs (1 with a functional issue), 2 test PRs (locked" in out


def test_bench_info_without_a_split_says_how_to_make_it(
    tiny_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Check that `bench info` still works before the split is made, and says how to make it."""
    monkeypatch.setattr("pr_reviewer.cli.split_path", lambda: tmp_path / "missing.json")
    assert main(["bench", "info", "--data", str(tiny_dataset)]) == 0
    out = capsys.readouterr().out
    assert "Split: not made yet (run: python scripts/make_split.py)" in out
    assert "pallets/flask           0      1      1\n" in out  # no dev and test columns


def test_bench_show_prints_the_review_input_and_answer_key(
    tiny_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that `bench show` prints the commits, the review input and the known issues."""
    assert main(["bench", "show", "psf__requests-5", "--data", str(tiny_dataset)]) == 0
    out = capsys.readouterr().out
    assert "psf__requests-5: psf/requests, opened 2021-03-01 12:00 UTC, functional, dev set\n" in (
        out
    )
    assert f"Head commit:  {sha(10)} (last of 1 reviewed commits)" in out
    assert "- **Title**: PR 5 \n" in out
    assert "1. E.2 Visual Representation: average_price divides" in out
    assert "2. F.4 Check: " in out


def test_bench_show_refuses_a_test_set_pr(
    tiny_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that `bench show` prints nothing of a test-set PR and exits 1."""
    assert main(["bench", "show", "psf__requests-2", "--data", str(tiny_dataset)]) == 1
    captured = capsys.readouterr()
    assert "Refused: psf__requests-2 is in the test set" in captured.err
    assert captured.out == ""


def test_bench_show_marks_a_pr_the_benchmark_doesnt_use(
    tiny_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that a PR outside the benchmark, such as one from 2018, can be shown."""
    assert main(["bench", "show", "psf__requests-1", "--data", str(tiny_dataset)]) == 0
    assert "functional, not used by the benchmark" in capsys.readouterr().out


def test_bench_show_unknown_id_explains(
    tiny_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that an unknown ID exits 1 and shows what IDs look like."""
    assert main(["bench", "show", "psf__requests-404", "--data", str(tiny_dataset)]) == 1
    assert "No PR with the ID 'psf__requests-404'" in capsys.readouterr().err


def test_bench_info_without_the_file_says_how_to_download(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that a missing dataset exits 1 with the download command."""
    assert main(["bench", "info", "--data", str(tmp_path / "none.jsonl")]) == 1
    assert "pr-reviewer bench download" in capsys.readouterr().err


def test_bench_download_skips_a_file_already_there(
    tiny_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that `bench download` sends nothing when the pinned file is already there."""
    assert main(["bench", "download", "--data", str(tiny_dataset)]) == 0
    assert "Already downloaded" in capsys.readouterr().out


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
