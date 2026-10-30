"""Write the two baseline reviews files for the dev set, made without a model.

- no-comments.jsonl approves every PR. Scored, it shows what doing nothing gets.
- known-issues.jsonl raises each PR's known issues, and approves clean PRs. Scored, it should
  get nearly every known issue; if not, the judge or the scoring is off.

The files are the same every time, so writing them again never changes a scored result. The
script prints the commands that score them (see src/pr_reviewer/benchmark/baselines.py).

How to run (inside the conda env, from the project folder, after `pr-reviewer bench download`):
    python scripts/write_baseline_reviews.py                 # writes into data/baselines/
    python scripts/write_baseline_reviews.py --out <folder>
"""

import argparse
import sys
from pathlib import Path

from pr_reviewer.benchmark.baselines import known_issue_reviews, no_comment_reviews
from pr_reviewer.benchmark.dataset import load_dataset
from pr_reviewer.benchmark.scoring import Prediction, save_predictions
from pr_reviewer.benchmark.source import DatasetError, default_dataset_path
from pr_reviewer.benchmark.split import SPLIT_SHA256, load_split, split_path, split_prs
from pr_reviewer.core.config import ConfigError

# Each baseline: its file name, the experiment name to score it under, and how it's made.
BASELINES = {
    "no-comments.jsonl": ("baseline-no-comments", no_comment_reviews),
    "known-issues.jsonl": ("baseline-known-issues", known_issue_reviews),
}


def run(out: Path | None) -> int:
    """Write both files and print how to score them.

    Args:
        out (Path | None): Folder to write into. None means data/baselines/ next to the
            dataset.

    Returns:
        int: Exit code, 0 on success.

    Raises:
        DatasetError: If the dataset or the split file is missing or isn't the pinned version.
        ConfigError: If the project folder can't be found.
    """
    data = default_dataset_path()
    folder = out or data.parent / "baselines"
    prs = split_prs(load_dataset(data), load_split(split_path(), sha256=SPLIT_SHA256), "dev")
    with_issues = sum(1 for pr in prs if pr.changes)
    print(f"Dev set: {len(prs)} PRs, {with_issues} with known issues.")
    commands = []
    for file_name, (experiment, make) in BASELINES.items():
        path = folder / file_name
        reviews: list[Prediction] = make(prs)
        save_predictions(path, reviews)
        shown = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
        print(f"Wrote {shown} ({len(reviews)} reviews)")
        commands.append(f"pr-reviewer score {shown} --name {experiment} --reviewer none")
    print("\nScore them with:\n  " + "\n  ".join(commands))
    return 0


def main() -> int:
    """Parse the arguments, run the script and turn setup problems into short messages.

    Returns:
        int: Exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--out", type=Path, default=None, help="folder to write into (default: data/baselines/)"
    )
    args = parser.parse_args()
    try:
        return run(args.out)
    except (DatasetError, ConfigError) as exc:
        print(f"Problem: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
