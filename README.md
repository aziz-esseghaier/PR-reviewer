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

## Models

`configs/models.yaml` lists the LLM providers and named model profiles. Code and experiments pick a model by its profile name. API keys stay in `.env`.

```
pr-reviewer llm models               # list the profiles and whether their keys are set
pr-reviewer llm available groq       # list the models your key can use at a provider
pr-reviewer llm ping <profile>       # send a short test prompt and show the reply and token counts
```

## Response cache

Every model reply is saved in PostgreSQL (table `llm_cache`). Asking the exact same thing again (same model, messages and settings) is answered from the database, so re-runs use no API quota and give the same answers. The cache never expires.

```
pr-reviewer llm ping <profile> --prompt "..."               # the second identical call is a cache hit
pr-reviewer llm ping <profile> --prompt "..." --sample 1    # ask again for a new answer (cached separately)
pr-reviewer llm ping <profile> --no-cache                   # skip the cache (works without the database)
pr-reviewer cache stats                                     # cached replies per provider and model
```

After pulling new code, run `pr-reviewer db migrate` to create any new tables.

## Checks

```
pytest                        # integration tests are skipped when the database isn't running
pytest -m "not integration"   # unit tests only
ruff check .
mypy
```
