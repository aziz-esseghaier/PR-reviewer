"""Tests for the connection string built in db/connection.py.

Checks that the POSTGRES_* settings end up in the connection string, that local defaults fill
in the host and port, and that a missing setting gives a clear error. No database needed.

How to run (from the project folder):
    pytest tests/db/test_db_connection.py
"""

import pytest
from psycopg.conninfo import conninfo_to_dict

from pr_reviewer.core.config import ConfigError
from pr_reviewer.db.connection import database_url

SETTINGS = {"POSTGRES_USER": "alice", "POSTGRES_PASSWORD": "s3cret word", "POSTGRES_DB": "reviews"}


def test_database_url_uses_settings_and_defaults() -> None:
    """Check that the settings end up in the connection string, with local defaults."""
    parts = conninfo_to_dict(database_url(env=SETTINGS))
    assert parts["host"] == "127.0.0.1"
    assert parts["port"] == "5432"
    assert parts["dbname"] == "reviews"
    assert parts["user"] == "alice"
    assert parts["password"] == "s3cret word"


def test_database_url_can_target_another_database() -> None:
    """Check that the dbname argument overrides POSTGRES_DB."""
    assert conninfo_to_dict(database_url(dbname="other", env=SETTINGS))["dbname"] == "other"


def test_database_url_reports_missing_settings() -> None:
    """Check that a missing password gives a clear error naming the setting."""
    with pytest.raises(ConfigError, match="POSTGRES_PASSWORD"):
        database_url(env={"POSTGRES_USER": "alice", "POSTGRES_DB": "reviews"})
