# PR reviewer

PR reviewer is an AI reviewer for pull requests in Python repos. It comments on a problem only when a test proves it. For each problem it suspects, it writes a pytest test and runs it on the code before and after the change. The problem counts as proved when the test passes before the change and fails after it with the error the reviewer predicted. The comment shows the problem, the proof and the test.

A review goes through these steps:

1. The setup agent installs the repo as it was when the pull request was opened. The install counts as working only when the repo's own tests pass.
2. The reviewer gathers the code around the change, and the warnings that four static tools (ruff, mypy, bandit and semgrep) give on the changed lines.
3. A model lists every problem that might be real.
4. For each problem, the model writes a test, and the differential runner runs it before and after the change.
5. The gate keeps only the problems a test proved, at most 3 per pull request.

The project also measures whether this rule makes reviews more precise. It runs the reviewer on [SWR-Bench](https://github.com/ZZR0/SWRench), a set of 1,000 real pull requests from 12 Python repos, and scores the reviews with SWR-Bench's own scorer and an LLM judge. Each experiment's settings, scores and summary are in its own folder under `results/`.

The reviewer uses `openai/gpt-oss-120b` on Groq's free tier, and the judge uses `gemini-3.5-flash-lite` on Google AI Studio. LangGraph runs the steps. PostgreSQL stores the results, every model reply and each run's progress, so a run stopped by a rate limit continues where it stopped.

Pull-request code is untrusted, so it runs only in Docker sandboxes with CPU, memory and time limits. While packages install, a sandbox can reach only an allow-list of sites, through a proxy. While tests run, its network is off.

A GitHub Action runs the same reviewer on live pull requests. It posts each proven problem as an inline comment and can also send a Slack message. It only comments, and never approves or merges.

## Project structure

```
PR-reviewer/
├── src/pr_reviewer/        the Python package
│   ├── cli.py              the pr-reviewer command
│   ├── core/               model client, reply cache, rate limits, usage log, tracing
│   ├── db/                 database connection, migrations, LangGraph checkpoints
│   ├── benchmark/          SWR-Bench data, dev and test sets, judge, scoring
│   ├── sandbox/            Docker sandboxes, network proxy, repo copies
│   ├── setup_agent/        builds and checks each PR's environment
│   ├── difftest/           runs tests before and after a PR and compares the results
│   ├── context/            finds the code around a change, the RAG part
│   ├── static/             runs ruff, mypy, bandit and semgrep on the changed lines
│   ├── reviewer/           the naive, context and candidate reviewers
│   ├── evidence/           checks each candidate with a generated test or a verifier
│   ├── gate/               keeps the proven findings, draws the precision-recall curve
│   ├── integrations/       GitHub, Slack and the review of a live pull request
│   └── demo_graph.py       a small LangGraph demo
├── configs/                models, sandbox limits, gate settings, dev and test sets
├── docker/                 Dockerfiles of the sandbox, proxy, indexer and tools images
├── scripts/                one-off checks: split, baselines, judge choice, proof fixes
├── tests/                  pytest tests, one folder per package
├── results/                each experiment's settings, scores and summary
├── third_party/swrbench/   SWR-Bench's official scoring scripts, unchanged
├── data/                   downloads, logs and generated files, not in git
├── action.yml              the GitHub Action
├── .github/workflows/      publishes the Docker images the Action uses
├── docker-compose.yml      PostgreSQL and pgAdmin
├── environment.yml         the conda environment
├── pyproject.toml          the package and its dependencies
└── .env.example            template for API keys and database settings
```

## How to run

Run the commands from the project folder, inside the conda environment. `pr-reviewer --help` lists the commands, and `pr-reviewer <command> --help` shows a command's options.

### Install

You need conda and Docker Desktop.

```
conda env create -f environment.yml
conda activate pr-reviewer
cp .env.example .env
```

Then fill in `.env`. The reviewer needs `GROQ_API_KEY`, and the judge needs `GEMINI_API_KEY`. Set `POSTGRES_PASSWORD` before the database's first start, because PostgreSQL reads it only when it creates its data.

When `pyproject.toml` gets new dependencies, run `pip install -e ".[dev]"` again.

### Database

```
docker compose up -d        # start PostgreSQL and pgAdmin
pr-reviewer db migrate      # create or update the tables
pr-reviewer db status       # show the server and the applied migrations
docker compose down         # stop them; the data stays in Docker volumes
```

To browse the tables, open pgAdmin at http://127.0.0.1:5050 and log in with `PGADMIN_EMAIL` and `PGADMIN_PASSWORD` from `.env`. Its first start takes about 10 minutes.

### Models

`configs/models.yaml` lists the model profiles and their rate limits. The client saves every reply in the database and answers a repeated request from there. When a provider answers with a rate-limit error, the client waits and tries again. When a daily quota runs out, the run stops with a message.

```
pr-reviewer llm models            # list the profiles and whether their keys are set
pr-reviewer llm ping <profile>    # send a short test prompt
pr-reviewer usage                 # quota use today, and requests per day
pr-reviewer cache stats           # saved replies per model
pr-reviewer demo <profile>        # run the small LangGraph demo
```

To see each run's steps and model calls in LangSmith, set `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true` in `.env`.

### Benchmark data

```
pr-reviewer bench download     # download SWR-Bench into data/ (85 MB)
pr-reviewer bench info         # count its PRs by repo, year and kind
pr-reviewer bench show <id>    # one PR's commits, the text the reviewer sees, its known issues
```

The project uses 433 of the PRs, from 2019 onward. A dev set of 130 PRs is for building the reviewer, and a test set of 303 PRs is for the final results only. The split is saved in `configs/swrbench_split.json`. Every command works on the dev set. The test set needs `--split test --unlock-test-set`, so no command uses it by accident.

### Environments

These steps make each PR ready to test. They use Docker and take hours the first time.

```
pr-reviewer sandbox build      # build the sandbox and proxy images (about 11 minutes)
pr-reviewer sandbox repos      # clone the PRs' repos into a Docker volume (about 1.6 GB)
pr-reviewer setup run          # build and check an environment for each dev PR from 2022 on
pr-reviewer setup report       # write the success rates to results/setup-dev/
pr-reviewer difftest prepare   # save each PR's environment, before and after the change, as images
```

Other useful commands:

```
pr-reviewer sandbox status             # images, repo copies and leftover containers
pr-reviewer sandbox cleanup            # remove containers left behind by a crash
pr-reviewer difftest run <id> <test>   # run a test before and after a PR, and compare
```

### Review the dev set

Run these in order. Each step saves its progress. If one stops, for example when the daily quota runs out, run it again to continue.

```
pr-reviewer context build        # find the code around each change (no model calls)
pr-reviewer static run           # run the static tools on the changed lines
pr-reviewer review candidates    # list possible problems, as run candidates-gpt-oss-120b
pr-reviewer evidence run         # check each one with a test or a verifier
pr-reviewer gate run --candidates candidates-gpt-oss-120b --evidence evidence-tools-gpt-oss-120b --name gated-dev-gpt-oss-120b --ungated-name ungated-dev-gpt-oss-120b
```

`gate run` writes two reviews. The gated one holds only the proven findings, and the ungated one holds every candidate. Score both with the judge, then write the comparison and the precision-recall curve:

```
pr-reviewer score results/gated-dev-gpt-oss-120b/raw/reviews.jsonl --name gated-dev-gpt-oss-120b --reviewer reviewer-gpt-oss-120b --since 2022
pr-reviewer score results/ungated-dev-gpt-oss-120b/raw/reviews.jsonl --name ungated-dev-gpt-oss-120b --reviewer reviewer-gpt-oss-120b --since 2022
pr-reviewer gate report --candidates candidates-gpt-oss-120b --evidence evidence-tools-gpt-oss-120b --name gated-dev-gpt-oss-120b --ungated-name ungated-dev-gpt-oss-120b
```

### Score and compare reviewers

`pr-reviewer score` scores a reviews file, one review per PR, with SWR-Bench's scorer and the judge. The results go to `results/<name>/`. Each review run writes its reviews to `results/<run>/raw/reviews.jsonl`, ready to score.

```
pr-reviewer score <reviews.jsonl> --name <name> --reviewer <profile> --since 2022
```

The simpler reviewers are there to compare against:

```
pr-reviewer review naive                              # one model call per PR, from the diff alone
pr-reviewer review context                            # the naive reviewer plus the code around the change
pr-reviewer review compare --base <run> --new <run>   # compare two scored runs, PR by PR
pr-reviewer evidence report                           # evidence tiers against the judge's verdicts
```

`evidence report` needs the candidate run scored first.

### Test-set run

The final results come from one run on the test set. Repeat the environment and review steps with `--split test --unlock-test-set` on every command. Also add `--usable-only` to `review candidates` and `score`, so only PRs with a working environment count, and add `--no-verifier` to `evidence run`, because the gate keeps only findings a test proved.

### GitHub Action

`action.yml` reviews a live pull request on GitHub's machines and posts what a test proved. Before the first use, run the "Publish images" workflow from this repo's Actions tab. It builds the 4 Docker images and publishes them to GitHub's registry. Then make each package public, so other repos can pull it.

In the repo to review, add the secret `GROQ_API_KEY`, and `SLACK_WEBHOOK_URL` if you want Slack messages. Then add this workflow:

```yaml
on:
  pull_request:
    types: [opened, reopened]
permissions:
  contents: read
  pull-requests: write
jobs:
  review:
    if: github.event.pull_request.head.repo.full_name == github.repository
    runs-on: ubuntu-24.04
    timeout-minutes: 120
    steps:
      - uses: <owner>/PR-reviewer@<commit>
        with:
          groq-api-key: ${{ secrets.GROQ_API_KEY }}
          slack-webhook-url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

The `if:` line skips pull requests from forks, because they get no secrets. The Action reviews pull requests without merge commits, in repos whose tests run with pytest.

To run the same review from your machine, start Docker and the database, and set `GITHUB_TOKEN`:

```
pr-reviewer action run --repo <owner/name> --pr <number> --dry-run   # print the comments instead of posting them
```

### Tests and checks

```
pytest                                        # all tests; database and Docker tests skip when those aren't running
pytest -m "not integration and not docker"    # unit tests only
pytest -m docker                              # sandbox tests, which need the images
ruff check .
mypy
```
