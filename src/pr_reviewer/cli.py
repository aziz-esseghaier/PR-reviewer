"""Command-line entry point for the PR reviewer."""

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import openai
import psycopg

from pr_reviewer import __version__
from pr_reviewer.core.config import ConfigError, load_env, load_models_config
from pr_reviewer.core.llm import LLMClient, LLMError, explain_api_error, list_provider_models
from pr_reviewer.db.connection import connect, extension_version, server_version
from pr_reviewer.db.migrate import applied_migrations, discover_migrations, migrate

Handler = Callable[[argparse.Namespace], int]

PING_PROMPT = "In one short sentence, what is a code review?"


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
    for migration in discover_migrations():
        applied_at = applied.get(migration.version)
        state = f"applied {applied_at:%Y-%m-%d %H:%M}" if applied_at else "pending"
        print(f"  {migration.version}_{migration.name}  {state}")
    return 0


def _llm_models(args: argparse.Namespace) -> int:
    """Print each model profile and whether its provider's API key is set.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `config`.

    Returns:
        int: Exit code, 0 on success.
    """
    config = load_models_config(args.config)
    for name, profile in config.models.items():
        key_env = config.provider(profile.provider).api_key_env
        key_state = "key set" if os.environ.get(key_env, "").strip() else f"{key_env} missing"
        print(
            f"{name:<24} {profile.provider:<8} {profile.model:<36} "
            f"temperature={profile.temperature} max_tokens={profile.max_tokens}  ({key_state})"
        )
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


def _llm_ping(args: argparse.Namespace) -> int:
    """Send one short prompt to a model profile and print the reply, tokens and timing.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `config`, `profile` and `prompt`.

    Returns:
        int: Exit code, 0 on success.
    """
    client = LLMClient(args.profile, load_models_config(args.config))
    response = client.complete([{"role": "user", "content": args.prompt}])
    print(f"Profile: {response.profile} ({response.provider}, {response.model})")
    print(f"Reply:   {response.text.strip() or '(empty)'}")
    print(f"Tokens:  {response.input_tokens} in, {response.output_tokens} out")
    print(f"Time:    {response.latency_s:.2f} s (finish reason: {response.finish_reason})")
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
    ping.set_defaults(handler=_llm_ping)
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
    try:
        return handler(args)
    except ConfigError as exc:
        print(f"Setup problem: {exc}", file=sys.stderr)
    except psycopg.OperationalError as exc:
        print(
            f"Can't reach PostgreSQL: {exc}\n"
            "Is Docker Desktop running? Start the database with: docker compose up -d",
            file=sys.stderr,
        )
    except openai.APIError as exc:
        print(explain_api_error(exc), file=sys.stderr)
    except LLMError as exc:
        print(f"Unusable model reply: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
