"""Command-line interface of the PR reviewer: the `pr-reviewer` command.

Parses the command line and runs the matching command. Setup, database and provider errors
are shown as a short hint with exit code 1, not as a traceback. Waits for rate limits and
retries are reported as they happen, on stderr. With LangSmith tracing on, a command waits
for its traces to upload before it exits.

How to run (inside the conda env, from the project folder):
    pr-reviewer --help                        list the commands
    pr-reviewer db migrate                    create or update the database tables
    pr-reviewer db status                     show the database and its migrations
    pr-reviewer llm models                    list the model profiles
    pr-reviewer llm available <provider>      list the models your key can use at a provider
    pr-reviewer llm ping <profile>            send a test prompt (answered from the cache if seen)
    pr-reviewer cache stats                   show how many replies the cache holds
    pr-reviewer usage                         show daily quota use and requests per day
    pr-reviewer demo <profile>                run the demo LangGraph graph (--thread to resume,
                                              --sample N for fresh answers)
    pr-reviewer bench download                download the pinned SWR-Bench dataset into data/
    pr-reviewer bench info                    count the benchmark's PRs by repo, year and kind
`python -m pr_reviewer.cli ...` works the same way.
"""

import argparse
import logging
import os
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import openai
import psycopg

from pr_reviewer import __version__
from pr_reviewer.benchmark.dataset import FIRST_YEAR, Instance, load_dataset, select_prs
from pr_reviewer.benchmark.source import (
    DATASET,
    SOURCE_COMMIT,
    SOURCE_REPO,
    DatasetError,
    check_official_files,
    default_dataset_path,
    download,
    matches_pin,
)
from pr_reviewer.core.cache import PostgresCache, cache_stats
from pr_reviewer.core.config import (
    ConfigError,
    describe_reset,
    load_env,
    load_models_config,
    read_api_key,
)
from pr_reviewer.core.limits import DAY, LimitError, daily_status
from pr_reviewer.core.llm import (
    ChatMessage,
    LLMClient,
    LLMError,
    LLMResponse,
    explain_api_error,
    list_provider_models,
)
from pr_reviewer.core.schemas import CATEGORY_NAMES
from pr_reviewer.core.tracing import describe_tracing, flush_traces, start_tracing
from pr_reviewer.core.usage import PostgresUsageLog, usage_by_day
from pr_reviewer.db.checkpoints import open_checkpointer, setup_checkpoints
from pr_reviewer.db.connection import connect, extension_version, server_version, table_exists
from pr_reviewer.db.migrate import applied_migrations, discover_migrations, migrate
from pr_reviewer.demo_graph import STATE_TYPES, DemoResult, new_thread_id, run_demo

Handler = Callable[[argparse.Namespace], int]

PING_PROMPT = "In one short sentence, what is a code review?"


class _StderrHandler(logging.Handler):
    """Writes log lines to sys.stderr as it is at that moment.

    A plain StreamHandler keeps the stream it started with. pytest swaps sys.stderr for each
    test, so that stream could be closed by the time a later test logs something.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Print one log line.

        Args:
            record (logging.LogRecord): The log entry to print.
        """
        print(self.format(record), file=sys.stderr)


def _show_progress_messages() -> None:
    """Print our own log lines (waits and retries) on stderr, once per process."""
    logger = logging.getLogger("pr_reviewer")
    if not any(isinstance(handler, _StderrHandler) for handler in logger.handlers):
        logger.addHandler(_StderrHandler())
        logger.setLevel(logging.INFO)


def _db_migrate(args: argparse.Namespace) -> int:
    """Apply new migrations and print which ones ran.

    Args:
        args (argparse.Namespace): Parsed command-line arguments (none are used).

    Returns:
        int: Exit code, 0 on success.
    """
    with connect() as conn:
        newly_applied = migrate(conn)
    if not newly_applied:
        print("Database is up to date. Nothing to apply.")
    for migration in newly_applied:
        print(f"Applied {migration.version}_{migration.name}")
    # LangGraph's checkpointer manages its own tables; its setup is safe to run every time.
    setup_checkpoints()
    print("LangGraph checkpoint tables are ready.")
    return 0


def _db_status(args: argparse.Namespace) -> int:
    """Print the database server, the pgvector version and the state of each migration.

    Args:
        args (argparse.Namespace): Parsed command-line arguments (none are used).

    Returns:
        int: Exit code, 0 on success.
    """
    with connect() as conn:
        target = f"{conn.info.host}:{conn.info.port}/{conn.info.dbname}"
        postgres = server_version(conn)
        pgvector = extension_version(conn, "vector")
        applied = applied_migrations(conn)
        checkpoints = table_exists(conn, "checkpoints")
    print(f"Database:    {target}")
    print(f"PostgreSQL:  {postgres}")
    print(f"pgvector:    {pgvector or 'not installed (run: pr-reviewer db migrate)'}")
    print(f"Checkpoints: {'ready' if checkpoints else 'missing (run: pr-reviewer db migrate)'}")
    print("Migrations:")
    migrations = discover_migrations()
    width = max((len(f"{m.version}_{m.name}") for m in migrations), default=0)
    for migration in migrations:
        applied_at = applied.get(migration.version)
        state = f"applied {applied_at:%Y-%m-%d %H:%M}" if applied_at else "pending"
        print(f"  {f'{migration.version}_{migration.name}':<{width}}  {state}")
    return 0


def _llm_models(args: argparse.Namespace) -> int:
    """Print each model profile, its settings and limits, and whether its API key is set.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `config`.

    Returns:
        int: Exit code, 0 on success.
    """
    config = load_models_config(args.config)
    for name, profile in config.models.items():
        key_env = config.provider(profile.provider).api_key_env
        key_state = "key set" if os.environ.get(key_env, "").strip() else f"{key_env} missing"
        rpm = profile.limits.requests_per_minute or "no limit"
        tpm = profile.limits.tokens_per_minute or "no limit"
        print(f"{name}  ({profile.provider}, {profile.model}, {key_state})")
        print(f"    temperature={profile.temperature}  max_tokens={profile.max_tokens}")
        print(f"    per minute: {rpm} requests, {tpm} tokens")
    return 0


def _llm_available(args: argparse.Namespace) -> int:
    """Print the model IDs a provider offers to your API key.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `config` and `provider`.

    Returns:
        int: Exit code, 0 on success.
    """
    config = load_models_config(args.config)
    for model_id in list_provider_models(config.provider(args.provider)):
        print(model_id)
    return 0


def _print_reply(response: LLMResponse, cache_used: bool) -> None:
    """Print a model reply with its tokens, timing and where it came from.

    Args:
        response (LLMResponse): The reply to print.
        cache_used (bool): Whether the cache was on for this call.
    """
    if not cache_used:
        source = "off (asked the provider, nothing saved)"
    elif response.cached:
        source = "hit (answered from the database, no request sent)"
    else:
        source = "miss (asked the provider and saved the reply)"
    original = " of the original call" if response.cached else ""
    print(f"Profile: {response.profile} ({response.provider}, {response.model})")
    print(f"Reply:   {response.text.strip() or '(empty)'}")
    print(f"Cache:   {source}")
    print(f"Tokens:  {response.input_tokens} in, {response.output_tokens} out{original}")
    print(
        f"Time:    {response.latency_s:.2f} s{original} (finish reason: {response.finish_reason})"
    )
    if not response.cached and (response.attempts > 1 or response.waited_s > 0):
        print(f"Waited:  {response.waited_s:.1f} s in total, {response.attempts} tries")


def _llm_ping(args: argparse.Namespace) -> int:
    """Send one prompt to a model profile and print the reply, using the cache unless told not to.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `config`, `profile`, `prompt`,
            `sample` and `no_cache`.

    Returns:
        int: Exit code, 0 on success.
    """
    config = load_models_config(args.config)
    # Check the profile name and its API key before opening the database, so a typo or a
    # missing key is reported as such, even when the database is down.
    profile = config.profile(args.profile)
    read_api_key(config.provider(profile.provider))
    messages: list[ChatMessage] = [{"role": "user", "content": args.prompt}]
    if args.no_cache:
        # No database at all: no cache, no usage log, no daily-limit check.
        response = LLMClient(args.profile, config).complete(messages, sample=args.sample)
        _print_reply(response, cache_used=False)
        return 0
    with connect() as conn:
        client = LLMClient(
            args.profile, config, cache=PostgresCache(conn), usage=PostgresUsageLog(conn)
        )
        response = client.complete(messages, sample=args.sample)
    _print_reply(response, cache_used=True)
    return 0


def _cache_stats(args: argparse.Namespace) -> int:
    """Print how many replies the cache holds per provider and model.

    Args:
        args (argparse.Namespace): Parsed arguments (none are used).

    Returns:
        int: Exit code, 0 on success.
    """
    with connect() as conn:
        stats = cache_stats(conn)
    if not stats:
        print("The cache is empty.")
        return 0
    print(f"{'PROVIDER':<10} {'MODEL':<36} {'REPLIES':>7} {'SIZE':>9}  NEWEST")
    for row in stats:
        size_kb = row.size_bytes / 1024
        print(
            f"{row.provider:<10} {row.model:<36} {row.replies:>7} {size_kb:>6.1f} kB  "
            f"{row.newest:%Y-%m-%d %H:%M}"
        )
    total = sum(row.replies for row in stats)
    print(f"Total: {total} {'reply' if total == 1 else 'replies'}")
    return 0


def _usage(args: argparse.Namespace) -> int:
    """Print each model's daily quota use now, then its requests per day.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `config` and `days`.

    Returns:
        int: Exit code, 0 on success.
    """
    config = load_models_config(args.config)
    now = datetime.now(UTC)
    with connect() as conn:
        log = PostgresUsageLog(conn)
        print("Daily quotas now (each provider's own reset rule)")
        seen: set[tuple[str, str]] = set()
        for profile in config.models.values():
            key = (profile.provider, profile.model)
            if key in seen:
                continue  # profiles on the same model share its quota
            seen.add(key)
            rule = config.provider(profile.provider).reset_rule
            history = log.recent(profile.provider, profile.model, now - DAY)
            limits = profile.limits
            requests = f"{len(history)} in the last 24 h"
            if limits.requests_per_day is not None:
                status = daily_status(
                    [(t, 1) for t, _ in history], limits.requests_per_day, rule, now
                )
                requests = f"{status.used} / {limits.requests_per_day}"
            tokens = f"{sum(tokens for _, tokens in history):,} in the last 24 h"
            if limits.tokens_per_day is not None:
                status = daily_status(history, limits.tokens_per_day, rule, now)
                tokens = f"{status.used:,} / {limits.tokens_per_day:,}"
            print(f"  {profile.provider} {profile.model}  ({describe_reset(rule)})")
            print(f"      requests: {requests}    tokens: {tokens}")
        days = usage_by_day(conn, args.days)
    print(f"\nBy day, last {args.days} days (UTC)")
    if not days:
        print("  No requests yet.")
        return 0
    header = f"  {'DAY':<10}  {'MODEL':<34} {'REQUESTS':>8} {'FAILED':>6} {'TOKENS':>9}"
    print(f"{header} {'CACHE HITS':>10} {'TOKENS SAVED':>12}")
    for row in days:
        model = f"{row.provider} {row.model}"
        print(
            f"  {row.day:%Y-%m-%d}  {model:<34} {row.requests:>8} {row.failed:>6} "
            f"{row.tokens:>9,} {row.cache_hits:>10} {row.tokens_saved:>12,}"
        )
    return 0


def _print_demo_result(result: DemoResult) -> None:
    """Print the claim, its tier and the review text of a demo run.

    Args:
        result (DemoResult): How the run ended.
    """
    if result.already_finished:
        print("  This thread had already finished, so nothing ran again. Its saved steps:")
        for line in result.log:
            print(f"  {line}")
    finding = result.finding
    if finding is not None:
        claim = finding.claim
        lines = f"{claim.start_line}-{claim.end_line}"
        category = f"{claim.category} {CATEGORY_NAMES[claim.category]}"
        print(f"Claim:  {claim.file}, lines {lines}, {category}, confidence {claim.confidence}")
        print(f"        {claim.summary}")
        print(f"Tier:   {finding.tier.value} ({finding.tier.name.lower().replace('_', ' ')})")
    print("Review text (what SWR-Bench would score):")
    for line in result.review.splitlines():
        print(f"  {line}")


def _demo(args: argparse.Namespace) -> int:
    """Run the demo LangGraph graph with one model profile, saving progress after each step.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `config`, `profile`, `thread` and
            `sample`.

    Returns:
        int: Exit code, 0 on success and 1 if a model call stopped the run.
    """
    config = load_models_config(args.config)
    profile = config.profile(args.profile)
    read_api_key(config.provider(profile.provider))
    thread_id = args.thread or new_thread_id(args.profile)
    print(f"Demo graph with {args.profile}, thread {thread_id}")
    print(f"Tracing: {describe_tracing()}")
    with connect() as conn, open_checkpointer(STATE_TYPES) as checkpointer:
        client = LLMClient(
            args.profile, config, cache=PostgresCache(conn), usage=PostgresUsageLog(conn)
        )
        try:
            result = run_demo(
                client,
                checkpointer,
                thread_id,
                on_step=lambda line: print(f"  {line}"),
                sample=args.sample,
            )
        except (openai.APIError, LimitError, LLMError) as exc:
            reason = explain_api_error(exc) if isinstance(exc, openai.APIError) else str(exc)
            print(
                f"Stopped: {reason}\nThe finished steps are saved. To continue, run:\n"
                f"  pr-reviewer demo {args.profile} --thread {thread_id}",
                file=sys.stderr,
            )
            return 1
    _print_demo_result(result)
    return 0


def _bench_download(args: argparse.Namespace) -> int:
    """Download the pinned SWR-Bench dataset, unless it's already there.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `data`.

    Returns:
        int: Exit code, 0 on success.
    """
    path: Path = args.data or default_dataset_path()
    if matches_pin(DATASET, path):
        print(f"Already downloaded: {path} (fingerprint matches)")
        return 0
    if path.exists():
        print(f"{path} isn't the pinned version, so it will be replaced.")
    size_mb = DATASET.size / 1_000_000
    source = f"{SOURCE_REPO} at {SOURCE_COMMIT[:7]}"
    print(f"Downloading {DATASET.name} ({size_mb:.0f} MB) from {source}")
    download(DATASET, path)
    print(f"Saved to {path} (fingerprint matches)")
    return 0


def _print_counts(title: str, prs: Sequence[Instance], key: Callable[[Instance], str]) -> None:
    """Print a table of functional and clean PRs per group, such as per repo or per year.

    Args:
        title (str): Header of the first column, such as "By repo".
        prs (Sequence[Instance]): The PRs to count.
        key (Callable[[Instance], str]): Gives the group of a PR, such as its repo.
    """
    counts = Counter((key(pr), pr.group) for pr in prs)
    rows = sorted({key(pr) for pr in prs})
    width = max(len(title), *(len(row) for row in rows))
    print(f"\n{title:<{width}}  FUNCTIONAL  CLEAN  TOTAL")
    for row in rows:
        functional, clean = counts[row, "functional"], counts[row, "clean"]
        print(f"{row:<{width}}  {functional:>10}  {clean:>5}  {functional + clean:>5}")


def _bench_info(args: argparse.Namespace) -> int:
    """Print where the benchmark data comes from and how its PRs break down.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `data`.

    Returns:
        int: Exit code, 0 on success.
    """
    path: Path = args.data or default_dataset_path()
    instances = load_dataset(path, sha256=DATASET.sha256)
    prs = select_prs(instances)
    with_issues = sum(instance.change_introduced for instance in instances)
    functional = [pr for pr in prs if pr.group == "functional"]
    mixed = sum(any(not issue.is_functional for issue in pr.changes) for pr in functional)
    issues = Counter(issue.is_functional for pr in functional for issue in pr.changes)
    older = sum(instance.year < FIRST_YEAR for instance in instances)
    only_e = sum(
        instance.year >= FIRST_YEAR and instance.group == "evolvability" for instance in instances
    )
    problems = check_official_files()

    print(f"Dataset:       {path}")
    print(f"Version:       {SOURCE_REPO} at {SOURCE_COMMIT[:7]} (fingerprint matches)")
    print(f"Scorer files:  {'; '.join(problems) or 'unchanged copies in third_party/swrbench/'}")
    print(f"All PRs:       {len(instances)} ({with_issues} with known issues)")
    print(f"\nBenchmark PRs: {len(prs)} from {FIRST_YEAR} onward, clean or with a functional issue")
    print(f"  {len(functional)} with a functional issue ({mixed} also have evolvability issues)")
    print(f"  {len(prs) - len(functional)} clean")
    print(f"  Known issues in them: {issues[True]} functional, {issues[False]} evolvability")
    print(
        f"Left out: {older} PRs from before {FIRST_YEAR}, and {only_e} from {FIRST_YEAR} "
        "onward with evolvability issues only"
    )
    _print_counts("By repo", prs, lambda pr: pr.repo)
    _print_counts("By year", prs, lambda pr: str(pr.year))
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the `pr-reviewer` command.

    Returns:
        argparse.ArgumentParser: Parser with the global options and subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="pr-reviewer",
        description="Evidence-first AI reviewer for Python pull requests.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(metavar="COMMAND")

    db = commands.add_parser("db", help="manage the PostgreSQL database")
    db_commands = db.add_subparsers(metavar="DB_COMMAND", required=True)
    db_commands.add_parser("migrate", help="apply new migrations").set_defaults(handler=_db_migrate)
    db_commands.add_parser(
        "status", help="show the server, the pgvector version and applied migrations"
    ).set_defaults(handler=_db_status)

    # Shared by every llm subcommand, so `--config` can follow the subcommand.
    config_option = argparse.ArgumentParser(add_help=False)
    config_option.add_argument(
        "--config",
        type=Path,
        default=None,
        help="model config file (default: configs/models.yaml in the project)",
    )
    llm = commands.add_parser("llm", help="list and try the configured LLMs")
    llm_commands = llm.add_subparsers(metavar="LLM_COMMAND", required=True)
    llm_commands.add_parser(
        "models", parents=[config_option], help="list the model profiles"
    ).set_defaults(handler=_llm_models)
    available = llm_commands.add_parser(
        "available", parents=[config_option], help="list the models a provider offers to your key"
    )
    available.add_argument("provider", help="provider name from the config, such as groq")
    available.set_defaults(handler=_llm_available)
    ping = llm_commands.add_parser(
        "ping", parents=[config_option], help="send a short test prompt to a model profile"
    )
    ping.add_argument("profile", help="model profile name from the config")
    ping.add_argument("--prompt", default=PING_PROMPT, help="prompt to send instead of the default")
    ping.add_argument(
        "--sample",
        type=int,
        default=0,
        help="ask again for a new answer to the same prompt: 1, 2, ... (default: 0)",
    )
    ping.add_argument(
        "--no-cache",
        action="store_true",
        help="skip the database: no cache, no usage log, no daily-limit check",
    )
    ping.set_defaults(handler=_llm_ping)

    cache = commands.add_parser("cache", help="inspect the LLM response cache")
    cache_commands = cache.add_subparsers(metavar="CACHE_COMMAND", required=True)
    cache_commands.add_parser(
        "stats", help="count the cached replies per provider and model"
    ).set_defaults(handler=_cache_stats)

    usage = commands.add_parser(
        "usage", parents=[config_option], help="show daily quota use and requests per day"
    )
    usage.add_argument("--days", type=int, default=7, help="days of history to show (default: 7)")
    usage.set_defaults(handler=_usage)

    demo = commands.add_parser(
        "demo", parents=[config_option], help="run the demo LangGraph graph with one model"
    )
    demo.add_argument("profile", help="model profile name from the config")
    demo.add_argument(
        "--thread",
        default=None,
        help="thread ID to resume a stopped run or show a finished one (default: a new thread)",
    )
    demo.add_argument(
        "--sample",
        type=int,
        default=None,
        help="ask for fresh answers instead of cached ones: 1, 2, ... (default: 0 for a new "
        "thread, the thread's own number when resuming)",
    )
    demo.set_defaults(handler=_demo)

    data_option = argparse.ArgumentParser(add_help=False)
    data_option.add_argument(
        "--data",
        type=Path,
        default=None,
        help="SWR-Bench dataset file (default: data/swr_datasets_d5c5.jsonl in the project)",
    )
    bench = commands.add_parser("bench", help="download and inspect the SWR-Bench data")
    bench_commands = bench.add_subparsers(metavar="BENCH_COMMAND", required=True)
    bench_commands.add_parser(
        "download", parents=[data_option], help="download the pinned SWR-Bench dataset into data/"
    ).set_defaults(handler=_bench_download)
    bench_commands.add_parser(
        "info", parents=[data_option], help="count the benchmark's PRs by repo, year and kind"
    ).set_defaults(handler=_bench_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line and run the chosen command.

    Args:
        argv (Sequence[str] | None): Arguments to parse. None means the real command line.

    Returns:
        int: Exit code, 0 on success and 1 on a setup, database, LLM provider or benchmark
        data problem.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    load_env()
    _show_progress_messages()
    start_tracing()
    try:
        return handler(args)
    except ConfigError as exc:
        print(f"Setup problem: {exc}", file=sys.stderr)
    except psycopg.OperationalError as exc:
        print(
            f"Can't reach PostgreSQL: {exc}\n"
            "Is Docker Desktop running? Start the database with: docker compose up -d\n"
            "(`llm ping` also works without the database: add --no-cache)",
            file=sys.stderr,
        )
    except openai.APIError as exc:
        print(explain_api_error(exc), file=sys.stderr)
    except LimitError as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
    except LLMError as exc:
        print(f"Unusable model reply: {exc}", file=sys.stderr)
    except DatasetError as exc:
        print(f"Benchmark data problem: {exc}", file=sys.stderr)
    finally:
        flush_traces()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
