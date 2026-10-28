# PR reviewer

An AI pull-request reviewer for Python repos. It posts a comment only when it can back the comment with evidence. SWR-Bench measures whether that rule improves precision.

## Setup

You need conda (Anaconda or Miniconda) and Docker Desktop. Conda installs Python 3.12.

```
conda env create -f environment.yml
conda activate pr-reviewer
cp .env.example .env          # then fill in your API keys
```

When `pyproject.toml` gets new dependencies, run `pip install -e ".[dev]"` again inside the environment.

## Database

PostgreSQL (with pgvector) runs in Docker. Set a real `POSTGRES_PASSWORD` in `.env` before the first start, because Postgres only reads it when it creates the data volume.

```
docker compose up -d          # start the database
pr-reviewer db migrate        # create or update the tables
pr-reviewer db status         # show the server, pgvector and applied migrations
docker compose down           # stop it (the data stays in the pgdata volume)
```

## Checks

```
pytest                        # integration tests are skipped when the database isn't running
pytest -m "not integration"   # unit tests only
ruff check .
mypy
```
