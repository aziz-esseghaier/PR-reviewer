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
docker compose up -d          # start the database and pgAdmin
pr-reviewer db migrate        # create or update the tables
pr-reviewer db status         # show the server, pgvector and applied migrations
docker compose down           # stop them (the data stays in the Docker volumes)
```

### Browse the tables (pgAdmin)

pgAdmin is a web tool for browsing the database, and it starts with it. Set `PGADMIN_EMAIL` and `PGADMIN_PASSWORD` in `.env` before its first start, because it only reads them then.

1. Open http://127.0.0.1:5050 and log in with those two values.
2. In the left panel, open Servers > pr-reviewer and enter your `POSTGRES_PASSWORD` (tick "Save password").
3. The tables are under Databases > (your `POSTGRES_DB`) > Schemas > public > Tables. Right-click one > View/Edit Data > All Rows.

The first start can take about 10 minutes while pgAdmin sets itself up. Later starts take under a minute. The page works once `docker compose ps` shows pgadmin as "healthy".

## Models

`configs/models.yaml` lists the LLM providers and named model profiles. Code and experiments pick a model by its profile name. API keys stay in `.env`.

```
pr-reviewer llm models               # list the profiles and whether their keys are set
pr-reviewer llm available groq       # list the models your key can use at a provider
pr-reviewer llm ping <profile>       # send a short test prompt and show the reply and token counts
```

## Rate limits and retries

Each profile in `configs/models.yaml` can set `limits` (`requests_per_minute`, `tokens_per_minute`). Within one run, the client waits before a request that would go over them. When a provider still answers 429 (rate limit), 5xx (overloaded) or doesn't answer in time, the client waits and tries again, up to 6 times. It uses the provider's own wait hint when there is one, and otherwise doubles the wait each time. A bad key or unknown model fails at once. If the provider asks for a wait over 2 minutes (usually a used-up daily quota), the run stops with a message instead of sleeping. Waits and retries are printed as they happen.

## Usage log and daily limits

Every request sent to a provider (each retry included) gets a row in the `llm_usage` table, and so does every reply served from the cache. Before each request, the client checks the model's `requests_per_day` and `tokens_per_day` against this log, counted the way the provider resets its quota (`daily_reset` in `configs/models.yaml`): all at once at a fixed time (Google: midnight Pacific), gradually over 24 hours (Groq), or over the last 24 hours when unknown. A short wait is waited out; a used-up quota stops the run with a message saying when it comes back.

```
pr-reviewer usage              # quota use now, and requests, failures, tokens and cache hits per day
pr-reviewer usage --days 30    # a longer history
```

Requests made outside this tool, for example in the AI Studio playground, aren't in the log.

## Response cache

Every model reply is saved in PostgreSQL (table `llm_cache`). Asking the exact same thing again (same model, messages and settings) is answered from the database, so re-runs use no API quota and give the same answers. The cache never expires.

```
pr-reviewer llm ping <profile> --prompt "..."               # the second identical call is a cache hit
pr-reviewer llm ping <profile> --prompt "..." --sample 1    # ask again for a new answer (cached separately)
pr-reviewer llm ping <profile> --no-cache                   # skip the cache (works without the database)
pr-reviewer cache stats                                     # cached replies per provider and model
```

After pulling new code, run `pr-reviewer db migrate` to create any new tables.

## Demo graph (LangGraph)

A small LangGraph graph reviews a built-in function with one model. It asks for one problem as JSON, and asks again (up to 3 times) when the reply doesn't fit the schema. Then it asks the same model to refute the problem, and prints the review text. The graph's progress is saved in PostgreSQL after each step (LangGraph's checkpoint tables, created by `pr-reviewer db migrate`), so a run stopped by a daily limit or an error continues where it stopped.

```
pr-reviewer demo <profile>                  # a new run (prints its thread ID)
pr-reviewer demo <profile> --sample 1       # a new run that asks for fresh answers (cached separately)
pr-reviewer demo <profile> --thread <id>    # resume a stopped run, or show a finished one
```

A second run on a new thread asks the same questions, so the cache answers all of them.

### End-to-end check

`scripts/check_foundations.py` runs the demo twice with each model in the config, using a sample number never used before. Run 1 must reach the model. Run 2 must send no request and take every reply from the cache. It prints the requests, cache hits and tokens of each run, then PASS or FAIL per model, and uses a few requests of each free daily quota.

```
python scripts/check_foundations.py                       # every profile in the config
python scripts/check_foundations.py groq-gpt-oss-120b     # only the profiles named
```

## Tracing (LangSmith)

LangSmith (smith.langchain.com) shows each run as a timeline: the graph's steps, and inside each step its model calls with the prompt, reply, tokens and time. Tracing is off until you turn it on:

1. Sign up for free at smith.langchain.com and create an API key (Settings > API Keys).
2. In `.env`, set `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true`. `LANGSMITH_PROJECT` names the project the traces go to.
3. Run `pr-reviewer demo <profile>`. It prints `Tracing: on (LangSmith project ...)`. Then open that project on the website.

A reply from the cache shows as a step marked `cached` with no tokens. A failed call shows its error. API keys are never sent. The free plan holds 5,000 traces a month, kept for 14 days, and one graph run is one trace. Tests never send traces.

## Benchmark data (SWR-Bench)

The benchmark is [SWR-Bench](https://github.com/ZZR0/SWRench): 1,000 real PRs from 12 Python repos, half with known issues (raised by the PR's real reviewers) and half clean. All its files come from one pinned commit of its repo, and each file is checked against a SHA-256 fingerprint before use, so the data can't change silently.

```
pr-reviewer bench download     # download the dataset (85 MB) into data/ (skipped when it's already there)
pr-reviewer bench info         # count the benchmark's PRs by repo, year and kind, and their input sizes
pr-reviewer bench show <id>    # one PR: its commits, the text the reviewer sees, and its known issues
```

The benchmark uses the PRs from 2019 onward that are clean or have at least one functional (`F.*`) issue: 433 PRs, 140 with a functional issue and 293 clean. PRs whose issues are all evolvability (`E.*`) ones are left out.

Each PR is reviewed as it stood when it was opened, before the reviewers' comments and the fixes that followed. SWR-Bench picks those commits by date, and loading the data re-checks every PR: a record whose reviewed commits include a fix, or don't match SWR-Bench's choice, is refused. The reviewer gets the PR's title, description and the diff of each reviewed commit, written exactly as SWR-Bench's own review script writes them (a test compares the two on all 1,000 PRs).

### Dev and test sets

The 433 PRs are split once into a dev set (130 PRs, used while building the reviewer) and a test set (303 PRs, used once for the final results). The split is stratified by repo, kind (functional or clean) and year, so each repo and each kind within it has about 30% of its PRs in dev. It's saved in `configs/swrbench_split.json`, and the code pins the file's fingerprint, so an edited split is refused.

The test set is locked: code that asks for test PRs gets an error unless it explicitly unlocks them, which only the final test-set run does. `bench show` refuses test-set PRs.

```
python scripts/make_split.py   # check that the saved split is what the seed gives (it never replaces the file)
```

### Judge model

SWR-Bench scores a review with an LLM judge, which splits the review into points and matches them to the PR's known issues. The judge is `gemini-3.5-flash-lite` (profile `judge-gemini-3.5-flash-lite`), fixed for the whole project so every result is scored the same way. The code refuses a judge profile whose model or settings changed, and that model isn't used as a reviewer. How it was chosen is in `results/judge-choice/decision.md`.

```
python scripts/compare_judges.py <model> ...   # compare candidate judges on test reviews with known right answers
```

### Scoring reviews

`pr-reviewer score` scores one review per dev PR with SWR-Bench's official scorer and the judge. The reviews file is JSONL, one line per PR, the format SWR-Bench's own review generator writes:

```
{"instance_id": "pydata__xarray-6710", "review": "1. xarray/core/dataset.py, line 12: ..."}
```

```
pr-reviewer score reviews.jsonl --name my-experiment --reviewer groq-gpt-oss-120b
pr-reviewer score reviews.jsonl --name a-baseline --reviewer none    # reviews written without a model
```

Before the first judge call it checks that every dev PR has exactly one review, that no review is for another PR, and that the reviews weren't written by the judge's own model. The test set is refused unless `--split test --unlock-test-set` is given, which only the final test-set run does. Judge replies are cached, so a run stopped by the daily quota continues where it stopped when run again, and scoring the same reviews again costs nothing. A dev run is about 130 judge calls.

Results go to `results/<name>/`: `config.json` (what was scored and how), `metrics.json` (SWR-Bench's metrics), `summary.md` (readable tables: PR-level accuracy and false alarms, then precision, recall and F1 of the review points, overall and for functional and evolvability issues apart), and `raw/` (the reviews and every judge reply, not in git). Per-PR counts also go to the `pr_scores` table, next to `bench_prs` (each PR's repo, year, kind and set), for grouping with SQL. Run `pr-reviewer db migrate` once to create those tables.

SWR-Bench's official scoring and review-generation scripts are copied unchanged into `third_party/swrbench/`, with SWR-Bench's MIT license. A test fails if they're edited.

## Checks

```
pytest                        # integration tests are skipped when the database isn't running
pytest -m "not integration"   # unit tests only
ruff check .
mypy
```
