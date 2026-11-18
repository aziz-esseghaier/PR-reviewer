"""
Tests for the dev/test split and the test-set lock in benchmark/split.py.

Builds small sets of PRs to check that the split is about 30/70 within every repo and kind,
that it doesn't depend on input order, that a saved split is never replaced and an edited one
is refused, and that test PRs stay locked. Two tests read the real files: the committed split
must match its pinned fingerprint, and once the dataset is downloaded, making the split again
from the seed must give the same split.

How to run (from the project folder):
    pytest tests/benchmark/test_split.py
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from pr_reviewer.benchmark.dataset import Instance, load_dataset, select_prs
from pr_reviewer.benchmark.source import DatasetError, sha256_of
from pr_reviewer.benchmark.split import (
    SPLIT_FILE,
    SPLIT_SHA256,
    LockedTestSetError,
    chosen_prs,
    ensure_unlocked,
    load_split,
    make_split,
    save_split,
    split_options,
    split_prs,
)
from tests.benchmark.test_dataset import REAL_DATASET, make_row

COMMITTED_SPLIT = Path(__file__).resolve().parents[2] / SPLIT_FILE


def many_prs() -> list[Instance]:
    """Make 80 PRs: three repos of different sizes, four years, functional and clean.

    Returns:
        list[Instance]: The PRs. Repo "org/big" has 48, "org/mid" 24 and "org/tiny" 8.
    """
    rows: list[dict[str, Any]] = []
    for repo, per_group in (("org/big", 6), ("org/mid", 3), ("org/tiny", 1)):
        for year in (2019, 2020, 2021, 2022):
            for issues in (("F.2 Logic",), ()):
                for _ in range(per_group):
                    rows.append(make_row(len(rows) + 1, year=year, issues=issues, repo=repo))
    return [Instance.model_validate_json(json.dumps(row)) for row in rows]


def test_split_is_30_70_within_every_repo_and_kind() -> None:
    """Check that every repo, and every kind within a repo, has within one PR of 30% in dev."""
    prs = many_prs()
    split = make_split(prs)
    dev = set(split.dev)
    assert len(split.dev) == 24
    assert len(split.dev) + len(split.test) == len(prs)
    for key in (lambda pr: pr.repo, lambda pr: (pr.repo, pr.group)):
        total = Counter(key(pr) for pr in prs)
        in_dev = Counter(key(pr) for pr in prs if pr.instance_id in dev)
        for group, size in total.items():
            assert abs(in_dev[group] - 0.3 * size) < 1, group


def test_split_ignores_the_input_order() -> None:
    """Check that reading the PRs in another order gives the same split."""
    prs = many_prs()
    assert make_split(prs) == make_split(list(reversed(prs)))


def test_seed_picks_the_order() -> None:
    """Check that another seed gives another split of the same size."""
    prs = many_prs()
    first, second = make_split(prs, seed=1), make_split(prs, seed=2)
    assert first.dev != second.dev
    assert len(first.dev) == len(second.dev)


def test_saved_split_reads_back_and_is_never_replaced(tmp_path: Path) -> None:
    """Check the round trip, the "\\n" line endings, and that a second save is refused."""
    split = make_split(many_prs())
    path = tmp_path / "split.json"
    save_split(split, path)
    assert load_split(path, sha256=None) == split
    assert b"\r" not in path.read_bytes()
    with pytest.raises(DatasetError, match="already exists"):
        save_split(make_split(many_prs(), seed=1), path)
    assert load_split(path, sha256=None) == split


def test_edited_split_is_refused(tmp_path: Path) -> None:
    """Check that moving one PR between the sets breaks the pinned fingerprint."""
    path = tmp_path / "split.json"
    save_split(make_split(many_prs()), path)
    pin = sha256_of(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["dev"].append(data["test"].pop())
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="was edited.*git checkout"):
        load_split(path, sha256=pin)


def test_split_with_a_pr_in_both_sets_is_invalid(tmp_path: Path) -> None:
    """Check that a PR can't be in the dev and the test set at once."""
    path = tmp_path / "split.json"
    save_split(make_split(many_prs()), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["dev"].append(data["test"][0])
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DatasetError, match="in both the dev and the test set"):
        load_split(path, sha256=None)


def test_missing_split_says_how_to_restore_it(tmp_path: Path) -> None:
    """Check that a missing split file points to git."""
    with pytest.raises(DatasetError, match="not found.*git checkout -- configs/swrbench_split"):
        load_split(tmp_path / "split.json")


def test_test_set_is_locked() -> None:
    """Check that test PRs come back only when the caller unlocks them."""
    prs = many_prs()
    split = make_split(prs)
    with pytest.raises(LockedTestSetError, match="locked until the final test-set run"):
        split_prs(prs, split, "test")
    unlocked = split_prs(prs, split, "test", unlock_test=True)
    assert [pr.instance_id for pr in unlocked] == [
        pr.instance_id for pr in prs if pr.instance_id in split.test
    ]
    assert len(split_prs(prs, split, "dev")) == len(split.dev)


def test_command_line_options_unlock_the_test_set_only_together() -> None:
    """Check --split test needs --unlock-test-set, and dev is the default."""
    prs = many_prs()
    split = make_split(prs)
    parser = split_options()
    assert len(chosen_prs(prs, split, parser.parse_args([]))) == len(split.dev)
    with pytest.raises(LockedTestSetError):
        chosen_prs(prs, split, parser.parse_args(["--split", "test"]))
    args = parser.parse_args(["--split", "test", "--unlock-test-set"])
    assert len(chosen_prs(prs, split, args)) == len(split.test)


def test_single_test_pr_is_locked() -> None:
    """Check ensure_unlocked: a test PR is refused, a dev PR or an unused PR is allowed."""
    split = make_split(many_prs())
    with pytest.raises(LockedTestSetError, match=f"{split.test[0]} is in the test set"):
        ensure_unlocked(split, split.test[0])
    ensure_unlocked(split, split.test[0], unlock_test=True)
    ensure_unlocked(split, split.dev[0])
    ensure_unlocked(split, "psf__requests-1")


def test_split_must_cover_exactly_the_benchmark_prs() -> None:
    """Check that a split missing a benchmark PR isn't used."""
    prs = many_prs()
    split = make_split(prs[1:])
    with pytest.raises(DatasetError, match="1 missing from it, 0 in it"):
        split_prs(prs, split, "dev")


def test_committed_split_matches_its_pin() -> None:
    """Check the committed split file: its pinned fingerprint and its 130/303 sizes."""
    assert sha256_of(COMMITTED_SPLIT) == SPLIT_SHA256
    split = load_split(COMMITTED_SPLIT)
    assert (len(split.dev), len(split.test)) == (130, 303)


def test_committed_split_is_what_the_seed_gives() -> None:
    """Check that making the split again from the real dataset gives the committed one."""
    if not REAL_DATASET.is_file():
        pytest.skip("SWR-Bench file not downloaded (run: pr-reviewer bench download)")
    prs = select_prs(load_dataset(REAL_DATASET))
    assert make_split(prs) == load_split(COMMITTED_SPLIT)
