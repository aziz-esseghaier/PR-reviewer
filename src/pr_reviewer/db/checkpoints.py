"""LangGraph checkpoints in PostgreSQL: saved graph progress, so a stopped run can resume.

After each step of a graph, LangGraph's checkpointer saves the graph's state under a thread ID.
Running the same thread again continues after the last finished step instead of starting over.
The checkpointer keeps its own tables (checkpoints, checkpoint_blobs, checkpoint_writes and
checkpoint_migrations), which `pr-reviewer db migrate` creates with the library's own setup.

When a checkpoint is read back, only the classes a graph lists are rebuilt as objects. Anything
else comes back as a plain dict, so a row in the database can't make the program build
arbitrary objects.

Imported by other modules; the file isn't run directly.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import DictRow, dict_row
from pydantic import BaseModel

from pr_reviewer.core.config import ConfigError
from pr_reviewer.db.connection import database_url, table_exists


def checkpoint_serializer(state_types: Sequence[type[BaseModel]]) -> JsonPlusSerializer:
    """Build the serializer that turns graph state into bytes for the database and back.

    Args:
        state_types (Sequence[type[BaseModel]]): The pydantic classes the graph keeps in its
            state, such as Finding. Only these are rebuilt as objects when read back.

    Returns:
        JsonPlusSerializer: LangGraph's serializer, limited to those classes.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=list(state_types))


def _connect(dbname: str | None) -> psycopg.Connection[DictRow]:
    """Open a connection set up the way the checkpointer needs it.

    Args:
        dbname (str | None): Database to connect to. None means the POSTGRES_DB setting.

    Returns:
        psycopg.Connection[DictRow]: Autocommit connection that returns rows as dicts.
    """
    # prepare_threshold=0 turns off server-side prepared statements, as the library does in
    # its own from_conn_string().
    return psycopg.connect(
        database_url(dbname), autocommit=True, prepare_threshold=0, row_factory=dict_row
    )


@contextmanager
def open_checkpointer(
    state_types: Sequence[type[BaseModel]] = (), dbname: str | None = None
) -> Iterator[PostgresSaver]:
    """Open a PostgreSQL checkpointer for a graph, and close its connection afterwards.

    Args:
        state_types (Sequence[type[BaseModel]]): The pydantic classes the graph keeps in its
            state, so they come back as objects.
        dbname (str | None): Database to use. None means the POSTGRES_DB setting.

    Yields:
        PostgresSaver: The checkpointer, to pass to a graph's compile().

    Raises:
        ConfigError: If the checkpoint tables don't exist yet.
    """
    with _connect(dbname) as conn:
        if not table_exists(conn, "checkpoints"):
            raise ConfigError("The checkpoint tables are missing. Run: pr-reviewer db migrate")
        yield PostgresSaver(conn, serde=checkpoint_serializer(state_types))


def setup_checkpoints(dbname: str | None = None) -> None:
    """Create or update the checkpointer's tables. Safe to run again.

    Args:
        dbname (str | None): Database to set up. None means the POSTGRES_DB setting.
    """
    with _connect(dbname) as conn:
        PostgresSaver(conn).setup()
