"""Open connections to the project's PostgreSQL database."""

import os
from collections.abc import Mapping

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg.rows import TupleRow

from pr_reviewer.core.config import ConfigError

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "5432"
# Fail fast when the container isn't running, instead of hanging.
CONNECT_TIMEOUT_SECONDS = 5
REQUIRED_SETTINGS = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")


def database_url(dbname: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """Build the connection string for the project's database from the POSTGRES_* settings.

    The string contains the password, so never print or log it.

    Args:
        dbname (str | None): Database to connect to. None means the POSTGRES_DB setting.
        env (Mapping[str, str] | None): Settings to read. None means the process environment.

    Returns:
        str: A libpq connection string, such as "host=127.0.0.1 port=5432 dbname=pr_reviewer ...".

    Raises:
        ConfigError: If a required setting is missing.
    """
    settings = os.environ if env is None else env
    missing = [name for name in REQUIRED_SETTINGS if not settings.get(name)]
    if missing:
        raise ConfigError(
            f"Missing settings: {', '.join(missing)}. Copy .env.example to .env and fill them in."
        )
    return make_conninfo(
        host=settings.get("POSTGRES_HOST") or DEFAULT_HOST,
        port=settings.get("POSTGRES_PORT") or DEFAULT_PORT,
        dbname=dbname or settings["POSTGRES_DB"],
        user=settings["POSTGRES_USER"],
        password=settings["POSTGRES_PASSWORD"],
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
    )


def connect(dbname: str | None = None) -> psycopg.Connection[TupleRow]:
    """Open an autocommit connection to the project's database.

    In autocommit mode each statement is saved on its own. Wrap statements that must succeed
    or fail together in `with conn.transaction():`.

    Args:
        dbname (str | None): Database to connect to. None means the POSTGRES_DB setting.

    Returns:
        psycopg.Connection[TupleRow]: An open connection. Use it in a `with` block so it closes.
    """
    return psycopg.connect(database_url(dbname), autocommit=True)


def server_version(conn: psycopg.Connection[TupleRow]) -> str:
    """Return the PostgreSQL server version.

    Args:
        conn (psycopg.Connection[TupleRow]): Open database connection.

    Returns:
        str: Version text, such as "18.1 (Debian 18.1-1.pgdg13+2)".
    """
    row = conn.execute("SHOW server_version").fetchone()
    return str(row[0]) if row else "unknown"


def extension_version(conn: psycopg.Connection[TupleRow], extension: str) -> str | None:
    """Return the installed version of a Postgres extension.

    Args:
        conn (psycopg.Connection[TupleRow]): Open database connection.
        extension (str): Extension name, such as "vector" for pgvector.

    Returns:
        str | None: The version, or None if the extension isn't installed in this database.
    """
    row = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = %s", (extension,)
    ).fetchone()
    return str(row[0]) if row else None
