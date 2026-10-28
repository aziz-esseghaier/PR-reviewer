"""Command-line interface of the PR reviewer: the `pr-reviewer` command.

Parses the command line and runs the matching command. Setup, database and provider errors
are shown as a short hint with exit code 1, not as a traceback. Waits for rate limits and
retries are reported as they happen, on stderr.

How to run (inside the conda env, from the project folder):
    pr-reviewer --help                        list the commands
    pr-reviewer db migrate                    create or update the database tables
    pr-reviewer db status                     show the database and its migrations
    pr-reviewer llm models                    list the model profiles
    pr-reviewer llm available <provider>      list the models your key can use at a provider
    pr-reviewer llm ping <profile>            send a test prompt (answered from the cache if seen)
    pr-reviewer cache stats                   show how many replies the cache holds
`python -m pr_reviewer.cli ...` works the same way.
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import openai
import psycopg

from pr_reviewer import __version__
from pr_reviewer.core.cache import PostgresCache, cache_stats
from pr_reviewer.core.config import ConfigError, load_env, load_models_config, read_api_key
from pr_reviewer.core.limits import LimitError
from pr_reviewer.core.llm import (
    ChatMessage,
    LLMClient,
    LLMError,
    LLMResponse,
    explain_api_error,
    list_provider_models,
)
from pr_reviewer.db.connection import connect, extension_version, server_version
from pr_reviewer.db.migrate import applied_migrations, discover_migrations, migrate

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
    print(f"Database:   {target}")
    print(f"PostgreSQL: {postgres}")
    print(f"pgvector:   {pgvector or 'not installed (run: pr-reviewer db migrate)'}")
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
        response = LLMClient(args.profile, config).complete(messages, sample=args.sample)
        _print_reply(response, cache_used=False)
        return 0
    with connect() as conn:
        client = LLMClient(args.profile, config, cache=PostgresCache(conn))
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
        help="skip the cache and always ask the provider (works without the database)",
    )
    ping.set_defaults(handler=_llm_ping)

    cache = commands.add_parser("cache", help="inspect the LLM response cache")
    cache_commands = cache.add_subparsers(metavar="CACHE_COMMAND", required=True)
    cache_commands.add_parser(
        "stats", help="count the cached replies per provider and model"
    ).set_defaults(handler=_cache_stats)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line and run the chosen command.

    Args:
        argv (Sequence[str] | None): Arguments to parse. None means the real command line.

    Returns:
        int: Exit code, 0 on success and 1 on a setup, database or LLM provider problem.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    load_env()
    _show_progress_messages()
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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
