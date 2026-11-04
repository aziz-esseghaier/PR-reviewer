"""The `pr-reviewer setup ...` commands: build environments for the dev PRs, and report on them.

cli.py adds them to the `pr-reviewer` command.

How to run (inside the conda env, from the project folder, with Docker Desktop and the
database running, after `pr-reviewer sandbox repos`):
    pr-reviewer setup run                   set up the dev PRs from 2022 onward not set up yet
    pr-reviewer setup run --since 2019      also the older ones
    pr-reviewer setup run --ids <id> ...    only these PRs
    pr-reviewer setup run --again ...       set PRs up again from the start
    pr-reviewer setup report                write results/setup-dev/ (rates, failures, subset)
                                            for the dev PRs from 2022 onward
    pr-reviewer setup report --since 2019   the same, with the older ones
"""

import argparse
import sys
from datetime import UTC, datetime

from pr_reviewer.benchmark.dataset import Instance, load_dataset
from pr_reviewer.benchmark.results import results_folder, save_bench_prs
from pr_reviewer.benchmark.source import DATASET, default_dataset_path
from pr_reviewer.benchmark.split import SPLIT_SHA256, Split, load_split, split_path, split_prs
from pr_reviewer.core.cache import PostgresCache
from pr_reviewer.core.config import load_models_config
from pr_reviewer.core.limits import LimitError
from pr_reviewer.core.llm import LLMClient
from pr_reviewer.core.schemas import MIN_PASS_RATE
from pr_reviewer.core.usage import PostgresUsageLog
from pr_reviewer.db.checkpoints import open_checkpointer
from pr_reviewer.db.connection import connect
from pr_reviewer.sandbox.commands import logs_dir, ready_image
from pr_reviewer.sandbox.config import load_sandbox_config
from pr_reviewer.sandbox.runner import docker_client
from pr_reviewer.setup_agent.agent import (
    MAX_MINUTES,
    MAX_MODEL_CALLS,
    SetupDeps,
    run_setup,
    thread_for,
)
from pr_reviewer.setup_agent.attempt import TryContext
from pr_reviewer.setup_agent.report import write_report
from pr_reviewer.setup_agent.store import latest_runs

# The setup agent's model profile in configs/models.yaml: one model for every setup.
SETUP_PROFILE = "setup-gpt-oss-120b"

# Dev PRs from this year onward are set up first.
FIRST_SETUP_YEAR = 2022


def _dev(args: argparse.Namespace) -> tuple[list[Instance], list[Instance], Split]:
    """Load the dataset, the split and the dev PRs.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `data`.

    Returns:
        tuple[list[Instance], list[Instance], Split]: All PRs, the dev PRs, and the split.
    """
    instances = load_dataset(args.data or default_dataset_path(), sha256=DATASET.sha256)
    split = load_split(split_path(), sha256=SPLIT_SHA256)
    return instances, split_prs(instances, split, "dev"), split


def _run(args: argparse.Namespace) -> int:
    """Set up the chosen dev PRs, one setup graph per PR, resuming any that were stopped.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `since`, `ids`, `limit`, `again`
            and `data`.

    Returns:
        int: Exit code, 0 when every chosen PR's setup finished (verified or not).
    """
    instances, dev, split = _dev(args)
    chosen = [pr for pr in dev if pr.year >= args.since]
    if args.ids:
        wanted = set(args.ids)
        chosen = [pr for pr in dev if pr.instance_id in wanted]
        if len(chosen) != len(wanted):
            missing = sorted(wanted - {pr.instance_id for pr in chosen})
            print(f"Not dev PRs (or unknown): {', '.join(missing)}", file=sys.stderr)
            return 1
    # Same repo and year together, so a PR finds the recipe the one before it made.
    chosen.sort(key=lambda pr: (pr.repo, pr.year, pr.created_at))
    client = docker_client()
    config = load_sandbox_config()
    image = ready_image(client, config)
    models = load_models_config()
    with connect() as conn, open_checkpointer() as checkpointer:
        save_bench_prs(conn, instances, split)
        done = latest_runs(conn)
        if not args.again:
            chosen = [pr for pr in chosen if pr.instance_id not in done]
        if args.limit:
            chosen = chosen[: args.limit]
        llm = LLMClient(
            SETUP_PROFILE, models, cache=PostgresCache(conn), usage=PostgresUsageLog(conn)
        )
        deps = SetupDeps(
            ctx=TryContext(client, config, image, logs_dir()),
            llm=llm,
            conn=conn,
            hosts=config.proxy.allowed_hosts,
        )
        print(f"Setting up {len(chosen)} dev PRs with {SETUP_PROFILE} (image {image})")
        verified = 0
        for number, pr in enumerate(chosen, start=1):
            print(f"[{number}/{len(chosen)}] {pr.instance_id} ({pr.year})", flush=True)
            thread = thread_for(pr, fresh=args.again)
            try:
                result = run_setup(
                    deps,
                    pr,
                    checkpointer,
                    thread,
                    on_step=lambda line: print(f"  {line}", flush=True),
                )
            except LimitError as exc:
                print(
                    f"Stopped: {exc}\nRun the same command again later: finished steps are "
                    "saved, and each setup continues where it stopped.",
                    file=sys.stderr,
                )
                return 1
            outcome = result.outcome
            verified += bool(outcome.get("verified"))
            verdict = (
                f"verified ({outcome.get('source')})"
                if outcome.get("verified")
                else f"failed: {outcome.get('failure_category')}"
            )
            print(
                f"  => {verdict}; {result.attempts} attempts, {result.model_calls} model calls, "
                f"{result.minutes:.0f} min",
                flush=True,
            )
    print(f"Done: {verified} of {len(chosen)} verified. Report: pr-reviewer setup report")
    return 0


def _report(args: argparse.Namespace) -> int:
    """Write results/setup-dev/ from the latest setup of each dev PR in scope.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `since` and `data`.

    Returns:
        int: Exit code, 0 on success.
    """
    _, dev, _ = _dev(args)
    prs = [pr for pr in dev if pr.year >= args.since]
    config = load_sandbox_config()
    with connect() as conn:
        runs = latest_runs(conn)
    settings = {
        "name": "setup-dev",
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "split": "dev",
        "since": args.since,
        "agent_model": SETUP_PROFILE,
        "agent_model_id": load_models_config().profile(SETUP_PROFILE).model,
        "budget": {"model_calls": MAX_MODEL_CALLS, "minutes": MAX_MINUTES},
        "min_pass_rate": MIN_PASS_RATE,
        "sandbox_limits": config.limits.model_dump(),
        "allowed_hosts": list(config.proxy.allowed_hosts),
    }
    folder = results_folder("setup-dev")
    usable = write_report(folder, prs, runs, settings)
    tried = sum(pr.instance_id in runs for pr in prs)
    print(
        f"{len(usable)} of {tried} set-up dev PRs from {args.since} onward verified. "
        f"Saved {folder} (summary.md)"
    )
    return 0


def _add_since(parser: argparse.ArgumentParser) -> None:
    """Add the `--since` option, which keeps only the PRs from a given year onward.

    Args:
        parser (argparse.ArgumentParser): The subcommand's parser.
    """
    parser.add_argument(
        "--since",
        type=int,
        default=FIRST_SETUP_YEAR,
        help=f"only PRs from this year onward (default: {FIRST_SETUP_YEAR})",
    )


def add_commands(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
    data_option: argparse.ArgumentParser,
) -> None:
    """Add `pr-reviewer setup ...` and its subcommands to the command line.

    Args:
        commands (argparse._SubParsersAction[argparse.ArgumentParser]): The top-level
            subcommands of `pr-reviewer`.
        data_option (argparse.ArgumentParser): Parent parser with the `--data` option.
    """
    setup = commands.add_parser("setup", help="build and check environments for the dev PRs")
    sub = setup.add_subparsers(metavar="SETUP_COMMAND", required=True)
    run = sub.add_parser("run", parents=[data_option], help="set up dev PRs")
    _add_since(run)
    run.add_argument("--ids", nargs="+", default=None, help="only these dev PRs")
    run.add_argument("--limit", type=int, default=None, help="at most this many PRs")
    run.add_argument(
        "--again", action="store_true", help="set PRs up again from the start, even if done"
    )
    run.set_defaults(handler=_run)
    report = sub.add_parser(
        "report", parents=[data_option], help="write results/setup-dev/ from the latest setups"
    )
    _add_since(report)
    report.set_defaults(handler=_report)
