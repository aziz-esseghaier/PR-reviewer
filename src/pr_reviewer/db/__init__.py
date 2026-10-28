"""Everything that talks to the project's PostgreSQL database.

- connection.py: builds the connection string from .env and opens connections.
- migrate.py: applies the numbered SQL files in migrations/ to create and update tables.
- migrations/: one SQL file per database change, applied in number order.

Imported by other modules. The database commands are `pr-reviewer db migrate` and
`pr-reviewer db status`.
"""
