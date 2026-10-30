"""Make the benchmark's dev/test split once, or check that the saved split is what the seed gives.

The first run splits the benchmark PRs about 30/70, stratified by repo, kind and year (see
src/pr_reviewer/benchmark/split.py), saves configs/swrbench_split.json and prints its
fingerprint to pin in split.py. Later runs never replace the file: they make the split again
and report whether it still matches the saved one. Prints dev and test counts per kind and repo.

How to run (inside the conda env, from the project folder, after `pr-reviewer bench download`):
    python scripts/make_split.py
"""

import sys
from collections import Counter

from pr_reviewer.benchmark.dataset import Instance, load_dataset, select_prs
from pr_reviewer.benchmark.source import DatasetError, default_dataset_path, sha256_of
from pr_reviewer.benchmark.split import (
    SPLIT_SHA256,
    Split,
    load_split,
    make_split,
    save_split,
    split_path,
)
from pr_reviewer.core.config import ConfigError


def print_balance(split: Split, prs: list[Instance]) -> None:
    """Print how many PRs of each repo and kind went to dev and to test.

    Args:
        split (Split): The split.
        prs (list[Instance]): The benchmark PRs.
    """
    counts = Counter((pr.repo, pr.group, split.which(pr.instance_id)) for pr in prs)
    width = max(len(pr.repo) for pr in prs)
    print(f"\n{'REPO':<{width}}  {'KIND':<10}  DEV  TEST  DEV SHARE")
    for repo, group in sorted({(pr.repo, pr.group) for pr in prs}):
        dev, test = counts[repo, group, "dev"], counts[repo, group, "test"]
        print(f"{repo:<{width}}  {group:<10}  {dev:>3}  {test:>4}  {dev / (dev + test):>8.0%}")


def run() -> int:
    """Make or check the split and print the outcome.

    Returns:
        int: Exit code, 0 when the split was saved or still matches.

    Raises:
        DatasetError: If the dataset is missing or isn't the pinned version.
        ConfigError: If the project folder can't be found.
    """
    prs = select_prs(load_dataset(default_dataset_path()))
    split = make_split(prs)
    path = split_path()
    if path.exists():
        saved = load_split(path, sha256=None)
        if saved != split:
            print(f"{path} differs from the split the seed gives now. Don't replace it.")
            return 1
        pinned = "pinned in split.py" if sha256_of(path) == SPLIT_SHA256 else "NOT pinned yet"
        print(f"{path} matches the split the seed gives (fingerprint {pinned}).")
    else:
        save_split(split, path)
        print(f"Saved {path}")
        print(f"Pin its fingerprint as SPLIT_SHA256 in split.py: {sha256_of(path)}")
    print(f"Dev: {len(split.dev)} PRs. Test: {len(split.test)} PRs.")
    print_balance(split, prs)
    return 0


def main() -> int:
    """Run the script and turn setup problems into short messages.

    Returns:
        int: Exit code, 0 on success.
    """
    try:
        return run()
    except (DatasetError, ConfigError) as exc:
        print(f"Problem: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
