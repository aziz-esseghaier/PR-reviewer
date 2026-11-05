"""The `pr-reviewer review ...` commands: review dev PRs with a reviewer, ready for scoring.

cli.py adds them to the `pr-reviewer` command. Only dev PRs are reviewed; the test set stays
locked until the final run.

How to run (inside the conda env, from the project folder, with the database running):
    pr-reviewer review naive                  review the dev PRs from 2022 on with the naive
                                              reviewer, as run "naive-gpt-oss-120b"
    pr-reviewer review naive --since 2019     the older dev PRs too (under another --name)
    pr-reviewer review naive --ids <id> ...   only these PRs (the run's other PRs stay to do)
    pr-reviewer review naive --limit 5        at most 5 PRs this time
Running it again continues a run: finished PRs are kept, a review stopped by the daily quota
resumes where it stopped, and failed PRs are tried again with fresh replies. Once every PR has
a review, it prints the `pr-reviewer score` command for the run.
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from typing import Any

from pr_reviewer.benchmark.dataset import Instance, load_dataset
from pr_reviewer.benchmark.judge import check_reviewer
from pr_reviewer.benchmark.results import results_folder, save_bench_prs
from pr_reviewer.benchmark.source import DATASET, default_dataset_path
from pr_reviewer.benchmark.split import SPLIT_SHA256, load_split, split_path, split_prs
from pr_reviewer.core.cache import PostgresCache
from pr_reviewer.core.config import load_models_config
from pr_reviewer.core.limits import LimitError
from pr_reviewer.core.llm import LLMClient
from pr_reviewer.core.usage import PostgresUsageLog
from pr_reviewer.db.checkpoints import open_checkpointer
from pr_reviewer.db.connection import connect
from pr_reviewer.reviewer.naive import (
    MAX_FINDINGS,
    MAX_TRIES,
    input_budget_chars,
    prompt_fingerprint,
    review_pr,
)
from pr_reviewer.reviewer.report import write_review_files
from pr_reviewer.reviewer.store import (
    load_findings,
    load_reviews,
    review_status,
    save_review,
    start_run,
)

# The reviewer's model profile in configs/models.yaml: one model for every reviewer experiment.
REVIEWER_PROFILE = "reviewer-gpt-oss-120b"

# The naive run's default name, which is also its results folder.
DEFAULT_NAME = "naive-gpt-oss-120b"

# Development starts on the dev PRs from this year on.
FIRST_YEAR = 2022


def _chosen(args: argparse.Namespace, in_scope: list[Instance]) -> list[Instance] | None:
    """Pick the PRs to review this time.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `ids`.
        in_scope (list[Instance]): The run's PRs.

    Returns:
        list[Instance] | None: The PRs asked for, or None (after a message) when one isn't in
        the run's scope.
    """
    if not args.ids:
        return in_scope
    wanted = set(args.ids)
    chosen = [pr for pr in in_scope if pr.instance_id in wanted]
    if len(chosen) != len(wanted):
        missing = sorted(wanted - {pr.instance_id for pr in chosen})
        print(
            f"Not dev PRs from {args.since} on (or unknown): {', '.join(missing)}",
            file=sys.stderr,
        )
        return None
    return chosen


def _naive(args: argparse.Namespace) -> int:
    """Review the chosen dev PRs with the naive reviewer, and write the run's folder.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `name`, `since`, `ids`, `limit`
            and `data`.

    Returns:
        int: Exit code, 0 when every PR in the run has a review, ready to score.
    """
    instances = load_dataset(args.data or default_dataset_path(), sha256=DATASET.sha256)
    split = load_split(split_path(), sha256=SPLIT_SHA256)
    in_scope = [pr for pr in split_prs(instances, split, "dev") if pr.year >= args.since]
    chosen = _chosen(args, in_scope)
    if chosen is None:
        return 1
    models = load_models_config()
    model = check_reviewer(REVIEWER_PROFILE, models)
    profile = models.profile(REVIEWER_PROFILE)
    budget = input_budget_chars(profile)
    folder = results_folder(args.name)
    settings: dict[str, Any] = {
        "name": args.name,
        "reviewer": "naive",
        "split": "dev",
        "since": args.since,
        "profile": REVIEWER_PROFILE,
        "provider": profile.provider,
        "model": model,
        "temperature": profile.temperature,
        "max_tokens": profile.max_tokens,
        "reasoning_effort": profile.reasoning_effort,
        "tokens_per_minute": profile.limits.tokens_per_minute,
        "input_budget_chars": budget,
        "max_findings": MAX_FINDINGS,
        "max_tries": MAX_TRIES,
        "prompt_sha256": prompt_fingerprint(),
        "dataset_sha256": DATASET.sha256,
        "split_sha256": SPLIT_SHA256,
    }
    # Part of every thread ID, so a saved review made with other settings is never resumed.
    settings_key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    stopped = ""
    with connect() as conn, open_checkpointer() as checkpointer:
        save_bench_prs(conn, instances, split)
        start_run(
            conn,
            args.name,
            reviewer="naive",
            profile=REVIEWER_PROFILE,
            model=model,
            split="dev",
            settings=settings,
        )
        status = review_status(conn, args.name)
        todo = [pr for pr in chosen if status.get(pr.instance_id, ("", 0))[0] != "done"]
        if args.limit:
            todo = todo[: args.limit]
        llm = LLMClient(
            REVIEWER_PROFILE, models, cache=PostgresCache(conn), usage=PostgresUsageLog(conn)
        )
        print(
            f"Reviewing {len(todo)} of the {len(in_scope)} dev PRs from {args.since} on with "
            f"{REVIEWER_PROFILE} (run {args.name}, up to {budget:,} characters of PR text)"
        )
        for number, pr in enumerate(todo, start=1):
            print(f"[{number}/{len(todo)}] {pr.instance_id}", flush=True)
            failures = status.get(pr.instance_id, ("", 0))[1]
            try:
                result = review_pr(
                    llm,
                    checkpointer,
                    pr,
                    f"{args.name}:{settings_key}:{pr.instance_id}:{failures}",
                    sample=failures,
                    budget_chars=budget,
                    on_step=lambda line: print(f"  {line}", flush=True),
                )
            except LimitError as exc:
                stopped = str(exc)
                break
            save_review(conn, args.name, result, failures + (0 if result.ok else 1))
            outcome = f"{len(result.findings)} findings" if result.ok else "failed"
            print(f"  => {outcome}; {result.tries} tries", flush=True)
        reviews = load_reviews(conn, args.name)
        findings = load_findings(conn, args.name)
    settings_file = {
        **settings,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "prs": len(in_scope),
    }
    predictions = write_review_files(folder, settings_file, in_scope, reviews, findings)
    if stopped:
        print(
            f"Stopped: {stopped}\nRun the same command again later: finished PRs are kept, "
            "and a stopped review continues where it stopped.",
            file=sys.stderr,
        )
        return 1
    if predictions is None:
        failed = sum(review.status == "failed" for review in reviews.values())
        waiting = len(in_scope) - sum(r.status == "done" for r in reviews.values())
        print(
            f"{waiting} PRs have no review yet ({failed} failed). Run the same command again: "
            f"failed PRs get fresh replies. Summary: {folder / 'review_summary.md'}"
        )
        return 1
    print(f"All {len(in_scope)} PRs reviewed. Saved {folder} (review_summary.md)")
    print(
        "Score it with:\n"
        f"  pr-reviewer score {predictions.relative_to(folder.parent.parent).as_posix()} "
        f"--name {args.name} --reviewer {REVIEWER_PROFILE} --since {args.since}"
    )
    return 0


def add_commands(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
    data_option: argparse.ArgumentParser,
) -> None:
    """Add `pr-reviewer review ...` and its subcommands to the command line.

    Args:
        commands (argparse._SubParsersAction[argparse.ArgumentParser]): The top-level
            subcommands of `pr-reviewer`.
        data_option (argparse.ArgumentParser): Parent parser with the `--data` option.
    """
    review = commands.add_parser("review", help="review dev PRs, ready for scoring")
    sub = review.add_subparsers(metavar="REVIEW_COMMAND", required=True)
    naive = sub.add_parser(
        "naive",
        parents=[data_option],
        help="one model call per PR, from the diff alone (the baseline)",
    )
    naive.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help=f"the run's name and results folder (default: {DEFAULT_NAME})",
    )
    naive.add_argument(
        "--since",
        type=int,
        default=FIRST_YEAR,
        help=f"only dev PRs from this year on (default: {FIRST_YEAR})",
    )
    naive.add_argument("--ids", nargs="+", default=None, help="only these PRs of the run")
    naive.add_argument("--limit", type=int, default=None, help="at most this many PRs this time")
    naive.set_defaults(handler=_naive)
