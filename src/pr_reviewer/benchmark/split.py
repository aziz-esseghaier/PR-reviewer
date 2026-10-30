"""The dev/test split of the benchmark PRs, and the lock on the test set.

The benchmark PRs are split about 30/70 into a dev set, used while building the reviewer, and a
test set, used once for the final results. The split is stratified by repo, kind (functional or
clean) and year: the PRs are sorted by those three, in a fixed pseudo-random order within each
group, and a running count of 30% sends a PR to dev each time it reaches a new whole number.
Any run of neighbors in that order, such as one repo's PRs or one repo's functional PRs, then
has within one PR of 30% in dev. Repo comes first so that every repo with more than one PR
has a dev PR, even django with four.

The pseudo-random order comes from a SHA-256 hash of the seed and each PR's ID, so the split
doesn't depend on the Python version or on the order the PRs are read in.

scripts/make_split.py makes the split once and saves it in configs/swrbench_split.json, which is
kept in git. Its SHA-256 is pinned here, so an edited file is refused.

The test set is locked: split_prs() and ensure_unlocked() refuse test PRs unless the caller
passes unlock_test=True, which only the final test-set run does.

Imported by other modules; the file isn't run directly.
"""

import hashlib
import math
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from pr_reviewer.benchmark.dataset import Instance, select_prs
from pr_reviewer.benchmark.source import DATASET, SOURCE_COMMIT, DatasetError, sha256_of
from pr_reviewer.core.config import project_root

DEV_FRACTION = Fraction(3, 10)
SPLIT_SEED = 2026
SPLIT_FILE = Path("configs") / "swrbench_split.json"
# Fingerprint of SPLIT_FILE as scripts/make_split.py saved it. The split never changes.
SPLIT_SHA256 = "902988f742372825324e3d042d64a8eeca4cc142fb9546e57dcb6084daffd3c2"

SPLIT_ABOUT = (
    "Dev/test split of the SWR-Bench benchmark PRs, stratified by repo, kind and year. Made "
    "once by scripts/make_split.py. Never edit it: its fingerprint is pinned in the code, and "
    "the test set stays locked until the final test-set run."
)

SplitName = Literal["dev", "test"]


class LockedTestSetError(DatasetError):
    """Raised when code asks for test-set PRs without unlocking them."""


class Split(BaseModel):
    """Which benchmark PRs are in the dev set and which are in the test set.

    Attributes:
        about (str): What the file is, for someone opening it.
        seed (int): Seed of the pseudo-random order within each group.
        dev_fraction (float): Share of the PRs in the dev set.
        source_commit (str): SWR-Bench commit the PRs come from.
        dataset_sha256 (str): Fingerprint of the dataset file the split was made from.
        dev (tuple[str, ...]): Instance IDs in the dev set, sorted.
        test (tuple[str, ...]): Instance IDs in the test set, sorted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    about: str = SPLIT_ABOUT
    seed: int
    dev_fraction: float
    source_commit: str
    dataset_sha256: str
    dev: tuple[str, ...]
    test: tuple[str, ...]

    @model_validator(mode="after")
    def _check_no_overlap(self) -> Self:
        """Make sure no PR is in both sets.

        Returns:
            Self: The unchanged split.

        Raises:
            ValueError: If an instance ID is in both sets.
        """
        if set(self.dev) & set(self.test):
            raise ValueError("a PR is in both the dev and the test set")
        return self

    def which(self, instance_id: str) -> SplitName | None:
        """Tell which set a PR is in.

        Args:
            instance_id (str): The PR's ID.

        Returns:
            SplitName | None: "dev" or "test", or None for a PR the benchmark doesn't use.
        """
        if instance_id in self.dev:
            return "dev"
        if instance_id in self.test:
            return "test"
        return None


def _hash_order(seed: int, text: str) -> str:
    """Give a text its place in the seed's pseudo-random order.

    Args:
        seed (int): The split's seed.
        text (str): Usually an instance ID.

    Returns:
        str: A SHA-256 hex digest. Sorting by it shuffles the texts the same way every time.
    """
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def make_split(
    prs: Sequence[Instance], seed: int = SPLIT_SEED, dev_fraction: Fraction = DEV_FRACTION
) -> Split:
    """Split the benchmark PRs into dev and test sets, stratified by repo, kind and year.

    Args:
        prs (Sequence[Instance]): The benchmark PRs, from select_prs().
        seed (int): Seed of the pseudo-random order within each group, and of the starting
            point of the running count.
        dev_fraction (Fraction): Share of the PRs that go to dev. A fraction, not a float, so
            the running count is exact.

    Returns:
        Split: The two sets.
    """
    ordered = sorted(
        prs, key=lambda pr: (pr.repo, pr.group, pr.year, _hash_order(seed, pr.instance_id))
    )
    # Where the running count starts, between 0 and 1, so the first PR isn't always in test.
    start = Fraction(int(_hash_order(seed, "start"), 16), 16**64)
    dev: list[str] = []
    test: list[str] = []
    for position, pr in enumerate(ordered):
        before = math.floor(start + position * dev_fraction)
        after = math.floor(start + (position + 1) * dev_fraction)
        (dev if after > before else test).append(pr.instance_id)
    return Split(
        seed=seed,
        dev_fraction=float(dev_fraction),
        source_commit=SOURCE_COMMIT,
        dataset_sha256=DATASET.sha256,
        dev=tuple(sorted(dev)),
        test=tuple(sorted(test)),
    )


def split_path() -> Path:
    """Where the split is kept: configs/swrbench_split.json in the project folder.

    Returns:
        Path: The file's path.

    Raises:
        ConfigError: If the project folder can't be found.
    """
    return project_root() / SPLIT_FILE


def save_split(split: Split, path: Path) -> None:
    """Save a split as JSON with one ID per line, never replacing an existing file.

    Lines end in "\\n" on every system, so the file's fingerprint is the same on Windows and
    Linux.

    Args:
        split (Split): The split to save.
        path (Path): Where to save it.

    Raises:
        DatasetError: If the file already exists.
    """
    if path.exists():
        raise DatasetError(f"{path} already exists. The split is made once and never replaced.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(split.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")


def load_split(path: Path, sha256: str | None = SPLIT_SHA256) -> Split:
    """Read the saved split and check that it wasn't edited.

    Args:
        path (Path): The split file.
        sha256 (str | None): Fingerprint the file must have. None skips the check, for test
            files and for checking a split before its fingerprint is pinned.

    Returns:
        Split: The split.

    Raises:
        DatasetError: If the file is missing, was edited, or isn't a valid split.
    """
    restore = f"Restore it from git: git checkout -- {SPLIT_FILE.as_posix()}"
    if not path.is_file():
        raise DatasetError(f"{path} not found. {restore}")
    if sha256 is not None and sha256_of(path) != sha256:
        raise DatasetError(f"{path} was edited (its fingerprint differs). {restore}")
    try:
        return Split.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise DatasetError(f"{path} isn't a valid split: {exc.errors()[0]['msg']}") from None


def check_split(split: Split, instances: Sequence[Instance]) -> None:
    """Make sure the split covers exactly the benchmark PRs of the dataset.

    Args:
        split (Split): The split.
        instances (Sequence[Instance]): All PRs from the dataset.

    Raises:
        DatasetError: If a benchmark PR is in neither set, or a set holds another PR.
    """
    expected = {pr.instance_id for pr in select_prs(instances)}
    actual = set(split.dev) | set(split.test)
    if actual != expected:
        raise DatasetError(
            f"The split doesn't match the benchmark PRs: {len(expected - actual)} missing "
            f"from it, {len(actual - expected)} in it that the benchmark doesn't use"
        )


def ensure_unlocked(split: Split, instance_id: str, *, unlock_test: bool = False) -> None:
    """Refuse a test-set PR unless the caller unlocked the test set.

    Args:
        split (Split): The split.
        instance_id (str): The PR about to be used.
        unlock_test (bool): True only for the final test-set run.

    Raises:
        LockedTestSetError: If the PR is in the test set and unlock_test is False.
    """
    if split.which(instance_id) == "test" and not unlock_test:
        raise LockedTestSetError(
            f"{instance_id} is in the test set, which stays locked until the final test-set "
            "run. Use a dev PR instead."
        )


def split_prs(
    instances: Sequence[Instance],
    split: Split,
    which: SplitName,
    *,
    unlock_test: bool = False,
) -> list[Instance]:
    """Pick the PRs of one set, in dataset order.

    Args:
        instances (Sequence[Instance]): All PRs from the dataset.
        split (Split): The split.
        which (SplitName): "dev" or "test".
        unlock_test (bool): True only for the final test-set run.

    Returns:
        list[Instance]: The PRs in that set.

    Raises:
        LockedTestSetError: If the test set is asked for without unlock_test.
        DatasetError: If the split doesn't match the dataset's benchmark PRs.
    """
    if which == "test" and not unlock_test:
        raise LockedTestSetError(
            "The test set stays locked until the final test-set run. Use the dev set."
        )
    check_split(split, instances)
    wanted = set(split.dev if which == "dev" else split.test)
    return [pr for pr in instances if pr.instance_id in wanted]
