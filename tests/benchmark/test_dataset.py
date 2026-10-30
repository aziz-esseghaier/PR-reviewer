"""
Tests for the SWR-Bench reader and PR selection in benchmark/dataset.py.

Most tests write a few small records shaped like SWR-Bench's. The last test reads the real
dataset and checks the counts the benchmark relies on; it is skipped until the file is
downloaded (`pr-reviewer bench download`).

How to run (from the project folder):
    pytest tests/benchmark/test_dataset.py
"""

import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from pr_reviewer.benchmark.dataset import Instance, load_dataset, select_prs
from pr_reviewer.benchmark.source import DATASET, DatasetError

REAL_DATASET = Path(__file__).resolve().parents[2] / "data" / DATASET.name


def sha(number: int) -> str:
    """Make a fake 40-character commit SHA.

    Args:
        number (int): Any number; different numbers give different SHAs.

    Returns:
        str: The SHA.
    """
    return f"{number:040x}"


def make_row(
    number: int,
    year: int = 2020,
    issues: Sequence[str] = ("F.2 Logic",),
    repo: str = "psf/requests",
) -> dict[str, Any]:
    """Make one dataset record shaped like SWR-Bench's, extra fields included.

    Args:
        number (int): PR number, used in the instance ID.
        year (int): Year the PR was opened.
        issues (Sequence[str]): Change types of its known issues, such as "F.2 Logic".
            Empty for a clean PR.
        repo (str): GitHub repo.

    Returns:
        dict[str, Any]: The record, ready to write as one JSON line.
    """
    commit = {
        "type": "commit",
        "sha": sha(2),
        "message": "Add average_price",
        "author": "someone",
        "date": f"{year}-03-01T12:00:00+00:00",
        "diff_text": "diff --git a/src/cart.py b/src/cart.py",
        "diff": [{"file": "src/cart.py", "patch": "@@ -1 +1,2 @@\n+def average_price(): ..."}],
    }
    changes = [
        {
            "change_type": change_type,
            "change_introducing": {"commit_sha": sha(2), "code_snippet": "def average_price()"},
            "change_discussion": {
                "discussion_summary": "average_price divides by zero for an empty cart.",
                "first_mention_timestamp": f"{year}-03-02T12:00:00+00:00",
                "original_reviewer_comment": "What if the cart is empty?",
            },
            "change_resolve_info": {
                "commit_sha": sha(3),
                "code_snippet": "if not items: return 0",
                "resolution_explanation": "Returns 0 for an empty cart.",
            },
        }
        for change_type in issues
    ]
    return {
        "repo": repo,
        "instance_id": f"{repo.replace('/', '__')}-{number}",
        "pr_title": f"PR {number}",
        "pr_statement": "Adds average_price.",
        "change_introduced": bool(issues),
        "base_commit": sha(1),
        "created_at": f"{year}-03-01T12:00:00+00:00",
        "changes": changes,
        "pr_commits": [commit],
        "pr_timeline": [{"type": "description", "user": "someone", "body": "Adds it."}],
        "all_commits": [commit, {**commit, "sha": sha(3), "message": "Handle empty carts"}],
    }


def write_dataset(folder: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Write records as a JSONL file, one per line.

    Args:
        folder (Path): Folder to write into.
        rows (Sequence[dict[str, Any]]): The records.

    Returns:
        Path: The file.
    """
    path = folder / "tiny.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def load(folder: Path, rows: Sequence[dict[str, Any]]) -> list[Instance]:
    """Write records to a file and read them back, without a fingerprint check.

    Args:
        folder (Path): Folder to write into.
        rows (Sequence[dict[str, Any]]): The records.

    Returns:
        list[Instance]: The checked PRs.
    """
    return load_dataset(write_dataset(folder, rows), sha256=None)


def test_record_is_read_with_its_commits_and_issues(tmp_path: Path) -> None:
    """Check the fields of one PR with a functional and an evolvability issue."""
    (pr,) = load(tmp_path, [make_row(7, issues=("F.2 Logic", "E.1.1 Textual Changes"))])
    assert pr.instance_id == "psf__requests-7"
    assert pr.year == 2020
    assert pr.pr_commits[0].diff[0].file == "src/cart.py"
    assert [issue.category for issue in pr.changes] == ["F.2", "E.1.1"]
    assert [issue.is_functional for issue in pr.changes] == [True, False]
    assert pr.changes[0].change_resolve_info is not None
    assert pr.changes[0].change_resolve_info.commit_sha == sha(3)
    assert pr.group == "functional"
    assert "pr_timeline" not in pr.model_dump()  # fields not used yet are skipped


def test_issue_without_a_fix_is_accepted(tmp_path: Path) -> None:
    """Check that change_resolve_info may be null, as it is for a few SWR-Bench issues."""
    row = make_row(1)
    row["changes"][0]["change_resolve_info"] = None
    (pr,) = load(tmp_path, [row])
    assert pr.changes[0].change_resolve_info is None


def test_selection_keeps_recent_clean_and_functional_prs(tmp_path: Path) -> None:
    """Check that old PRs and PRs with only evolvability issues are left out."""
    rows = [
        make_row(1, year=2018),  # too old
        make_row(2, year=2019),  # functional
        make_row(3, year=2019, issues=("E.3.2 Solution Approach",)),  # evolvability only
        make_row(4, year=2020, issues=()),  # clean
        make_row(5, year=2021, issues=("E.2 Visual Representation", "F.4 Check")),  # both
    ]
    instances = load(tmp_path, rows)
    assert [pr.group for pr in instances] == [
        "functional",
        "functional",
        "evolvability",
        "clean",
        "functional",
    ]
    assert [pr.instance_id[-1] for pr in select_prs(instances)] == ["2", "4", "5"]


def test_year_is_counted_in_utc(tmp_path: Path) -> None:
    """Check that a PR opened late on New Year's Eve west of UTC counts for the next year."""
    row = make_row(1, year=2018)
    row["created_at"] = "2018-12-31T23:30:00-02:00"  # 01:30 UTC on 1 January 2019
    (pr,) = load(tmp_path, [row])
    assert pr.year == 2019


def _unknown_category(row: dict[str, Any]) -> None:
    row["changes"][0]["change_type"] = "X.1 Mystery"


def _flag_mismatch(row: dict[str, Any]) -> None:
    row["change_introduced"] = False


def _short_sha(row: dict[str, Any]) -> None:
    row["base_commit"] = "abc123"


def _no_commits(row: dict[str, Any]) -> None:
    row["pr_commits"] = []


@pytest.mark.parametrize(
    ("spoil", "message"),
    [
        (_unknown_category, "changes.0.change_type: Value error, must start with a category"),
        (_flag_mismatch, "change_introduced doesn't match"),
        (_short_sha, "base_commit: String should match pattern"),
        (_no_commits, "pr_commits: Tuple should have at least 1 item"),
    ],
)
def test_bad_record_is_reported_with_its_line(
    tmp_path: Path, spoil: Callable[[dict[str, Any]], None], message: str
) -> None:
    """Check that a record that doesn't fit is rejected, naming its line and the field."""
    bad = make_row(2)
    spoil(bad)
    with pytest.raises(DatasetError, match=f"tiny.jsonl, line 2: .*{message}"):
        load(tmp_path, [make_row(1), bad])


def test_repeated_instance_is_rejected(tmp_path: Path) -> None:
    """Check that the same instance ID twice is an error, not a silent double count."""
    with pytest.raises(DatasetError, match="line 2: psf__requests-1 repeats"):
        load(tmp_path, [make_row(1), make_row(1)])


def test_other_version_of_the_file_is_refused(tmp_path: Path) -> None:
    """Check that a file without the pinned fingerprint is refused, with how to fix it."""
    path = write_dataset(tmp_path, [make_row(1)])
    with pytest.raises(DatasetError, match="fingerprint differs.*pr-reviewer bench download"):
        load_dataset(path)


def test_missing_file_says_how_to_download(tmp_path: Path) -> None:
    """Check that a missing dataset points to the download command."""
    with pytest.raises(DatasetError, match="not found.*pr-reviewer bench download"):
        load_dataset(tmp_path / "missing.jsonl")


def test_real_dataset_has_the_expected_prs() -> None:
    """Check the real file: 1,000 PRs, and 433 benchmark PRs (140 functional, 293 clean)."""
    if not REAL_DATASET.is_file():
        pytest.skip("SWR-Bench file not downloaded (run: pr-reviewer bench download)")
    instances = load_dataset(REAL_DATASET)
    assert len(instances) == 1000
    assert Counter(instance.change_introduced for instance in instances) == {True: 500, False: 500}
    prs = select_prs(instances)
    assert Counter(pr.group for pr in prs) == {"functional": 140, "clean": 293}
    assert len({pr.repo for pr in prs}) == 12
