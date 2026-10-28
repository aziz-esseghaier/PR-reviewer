"""Command-line entry point for the PR reviewer."""

import argparse
import sys
from collections.abc import Callable, Sequence

import psycopg

from pr_reviewer import __version__
from pr_reviewer.core.config import ConfigError, load_env
from pr_reviewer.db.connection import connect, extension_version, server_version
from pr_reviewer.db.migrate import applied_migrations, discover_migrations, migrate

Handler = Callable[[argparse.Namespace], int]


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line and run the chosen command.

    Args:
        argv (Sequence[str] | None): Arguments to parse. None means the real command line.

    Returns:
        int: Exit code, 0 on success and 1 on a setup or connection problem.
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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
