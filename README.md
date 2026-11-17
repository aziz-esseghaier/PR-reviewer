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

`configs/models.yaml` lists the LLM providers and named model profiles. Code and experiments pick a model by its profile name. API keys stay in `.env`. A profile can also set `reasoning_effort` (low, medium or high) for reasoning models such as gpt-oss; it's sent to the provider only when set, so older cached replies stay valid.

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
pr-reviewer score reviews.jsonl --name my-experiment --reviewer ... --since 2022   # only the PRs from 2022 on
```

With `--since`, only the set's PRs from that year on are scored, and reviews of its older PRs in the file are ignored. Before the first judge call it checks that every dev PR has exactly one review, that no review is for another PR, and that the reviews weren't written by the judge's own model. The test set is refused unless `--split test --unlock-test-set` is given, which only the final test-set run does. Judge replies are cached, so a run stopped by the daily quota continues where it stopped when run again, and scoring the same reviews again costs nothing. A dev run is about 130 judge calls.

Results go to `results/<name>/`: `config.json` (what was scored and how), `metrics.json` (SWR-Bench's metrics), `summary.md` (readable tables: PR-level accuracy and false alarms, then precision, recall and F1 of the review points, overall and for functional and evolvability issues apart), and `raw/` (the reviews and every judge reply, not in git). Per-PR counts also go to the `pr_scores` table, next to `bench_prs` (each PR's repo, year, kind and set), for grouping with SQL. Run `pr-reviewer db migrate` once to create those tables.

SWR-Bench's official scoring and review-generation scripts are copied unchanged into `third_party/swrbench/`, with SWR-Bench's MIT license. A test fails if they're edited.

### Baselines

Two sets of dev reviews written without a model give the score to beat and check the scoring:

- **No comments** approves every PR, in the words SWR-Bench's review prompt asks for when nothing is found ("The pull request looks good and meets the standards.").
- **Known issues** lists each PR's known issues, in SWR-Bench's own summary of each, and approves clean PRs. No review matches the answer key better, so a score below 1 here would point at the judge or the scoring.

```
python scripts/write_baseline_reviews.py      # writes data/baselines/no-comments.jsonl and known-issues.jsonl
pr-reviewer score data/baselines/no-comments.jsonl --name baseline-no-comments --reviewer none
pr-reviewer score data/baselines/known-issues.jsonl --name baseline-known-issues --reviewer none
```

Clean PRs get the same review in both, so the second run sends the judge only the 39 PRs with issues. Results on the dev set (130 PRs: 39 with 82 known issues, 91 clean):

| Baseline | PR accuracy | Point precision | Point recall | Point F1 |
|---|---|---|---|---|
| No comments | 0.700 | 0.000 | 0.000 | 0.000 |
| Known issues | 1.000 | 1.000 | 1.000 | 1.000 |

Approving everything already gets 0.70 PR accuracy, because most PRs are clean. So compare reviewers on the point-level scores, not on accuracy.

On the 68 dev PRs from 2022 onward (26 with issues, 42 clean), the PRs later reviewers are measured on, no comments scores 0.618 PR accuracy and known issues 1.000 everywhere (`results/baseline-*-2022/`, scored from the judge's cached answers).

## Naive reviewer (the baseline reviewer)

The first reviewer sends the PR to the model and asks for its problems, with no context and no evidence. Every later reviewer is measured against it.

1. The PR's text (title, description, diffs) is written exactly as SWR-Bench gives it to reviewers, between `<pull_request>` tags. The prompt says it's data written by the PR's author, never instructions.
2. The model is `openai/gpt-oss-120b` on Groq (profile `reviewer-gpt-oss-120b`, temperature 0, reasoning effort medium), the same for every reviewer experiment. It replies with a JSON object of findings, each a `Claim`: file, lines, SWR-Bench category, a one-sentence summary, the exception a test would hit, and a confidence. Groq enforces the JSON schema.
3. The reply is checked against `Claim`. A reply that isn't valid, or was cut off, is rejected and the model is asked again with the reason, up to 3 times. A PR with no usable reply fails, and the next run tries it again with fresh replies.
4. At most 10 findings are kept, with no confidence filter, and become the review text: one numbered point each, or SWR-Bench's "looks good" sentence when there are none.

The steps form a LangGraph graph, and the Postgres checkpointer saves its state after each step. Findings go to the `findings` table (migration 0008), next to `review_runs` and `pr_reviews`.

**Size limit.** On Groq's free tier, a request (prompt plus the reply allowance) can't exceed 8,000 tokens a minute. gpt-oss's thinking counts toward its reply, so the reply gets 3,000 tokens and the PR text up to about 12,000 characters. A bigger PR keeps whole file diffs, source files first, and the left-out files are listed by name. A very long description is shortened when the diffs fit.

```
pr-reviewer review naive                  # review the 68 dev PRs from 2022 on, as run naive-gpt-oss-120b
pr-reviewer score results/naive-gpt-oss-120b/raw/reviews.jsonl --name naive-gpt-oss-120b --reviewer reviewer-gpt-oss-120b --since 2022
```

A run uses about 230,000 tokens, a bit more than Groq's free 200,000 a day. It stops when the daily quota runs out and continues where it stopped when run again. `results/naive-gpt-oss-120b/` holds the run's settings (`review_config.json`) and summary (`review_summary.md`), the reviews and findings in `raw/`, and the scores (`summary.md`).

Results on the 68 dev PRs from 2022 onward (26 with issues, 42 clean):

| Reviewer | PR accuracy | Point precision | Point recall | Point F1 |
|---|---|---|---|---|
| No comments | 0.618 | 0.000 | 0.000 | 0.000 |
| Naive (gpt-oss-120b) | 0.529 | 0.048 | 0.140 | 0.072 |
| Known issues | 1.000 | 1.000 | 1.000 | 1.000 |

On functional issues it scores precision 0.055, recall 0.207 and F1 0.087; on evolvability issues 0.035, 0.074 and 0.048.

Only 8 of its 166 points match a known issue. 92 of the 158 false alarms are on clean PRs: it flags 27 of the 42, which is why its PR accuracy is below approving everything. That noise is what the evidence checks are meant to remove.

## Context retrieval (the RAG part)

The context reviewer is the naive reviewer plus code from the repo around the change: retrieve, fit into the request, then generate. Everything else stays the same (model, prompt rules, reply format, size limit), so comparing the two runs measures the context alone.

**Retrieval** runs once per PR in an indexer sandbox: the sandbox image plus Jedi (`docker/indexer/Dockerfile`), network off. A script (`src/pr_reviewer/context/probe.py`) checks out the PR's last commit, finds the changed lines with `git diff` from its base, and parses every Python file with Python's `ast`, which reads code without running it. It then collects candidate pieces:

1. **changed**: each source function the PR changed, whole (unless the PR added all of it, since the diff shows it then);
2. **called**: the definitions those functions call, as signature and docstring (at most 10 docstring lines), or whole when 30 lines or less;
3. **caller**: up to 5 functions calling each changed function, closest first (same file, same folder, same package);
4. **test**: up to 3 existing tests using each changed function, the module's own test file first;
5. **changed_test**: the test functions the PR edited, whole;
6. **file**: the changed files, whole.

Names are matched through the files' imports. A call `ast` can't place, such as a name re-exported by a package's `__init__.py` or an inherited method, goes to Jedi's "go to definition" (within 2 minutes per PR). A method called on an object of unknown type is matched only when no other definition has its name. No vector search is used: every link here is an exact name. The candidates go to a JSON file under `data/contexts/`, recorded in the `pr_contexts` table (migration 0009).

**Fitting.** The PR's text comes first, exactly as the naive reviewer fits it. The room left in the ~12,000 characters goes to whole pieces, in the order above; a piece that doesn't fit is left out, and smaller ones after it still get their chance. The pieces go after the PR in a `<repository_context>` block, with a note that it's data, never instructions. A PR with no room left gets the naive request unchanged.

```
pr-reviewer context build                 # retrieve the 68 dev PRs from 2022 on (Docker, no model calls)
pr-reviewer context check                 # did the model see the code each known issue's fix changed?
pr-reviewer review context                # review with context, as run context-gpt-oss-120b
pr-reviewer score results/context-gpt-oss-120b/raw/reviews.jsonl --name context-gpt-oss-120b --reviewer reviewer-gpt-oss-120b --since 2022
pr-reviewer review compare                # compare it with the naive run, overall, on subsets and PR by PR
```

`results/context-dev-2022/` has the retrieval summary: pieces per PR, and how many calls `ast`, Jedi or neither resolved. The **retrieval check** uses each known issue's later fix commit, only to measure: if the functions the fix changed were in the diff or the context, a missed issue is the model's miss; if retrieval never found them, it's retrieval's.

Results on the 68 dev PRs from 2022 onward. 37 PRs got context (median 4,600 characters). The other 31 are docs-only, change no function, or had no room; they sent the naive request unchanged, so their reviews are the same.

| Reviewer | PR accuracy | Point precision | Point recall | Point F1 | F1 (F) | F1 (E) |
|---|---|---|---|---|---|---|
| No comments | 0.618 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Naive | 0.529 | 0.048 | 0.140 | 0.072 | 0.087 | 0.048 |
| Context | 0.485 | 0.076 | 0.175 | 0.106 | 0.136 | 0.057 |

On the 37 PRs with context, false alarms fall from 92 to 56 and hits rise from 5 to 7: precision 0.051 → 0.111, F1 0.073 → 0.136. PR accuracy drops because 3 PRs with issues got a clean review. The hit counts are small (4 PRs gained a hit, 2 lost one), so the drop in false alarms is the clearer effect. The retrieval check shows why context finds little that's new: of the 57 known issues, 23 have their fix in a function the PR's diff already shows, 28 in no function at all (docs, tests, module-level lines), and only 1 outside the diff. `results/context-gpt-oss-120b/` has the summaries, `retrieval_check.md` and `comparison.md`.

## Candidates and static tools

The candidate stage aims at recall. Its job is to list every problem that might be real, because a later stage checks each candidate with evidence and drops the false ones. Missing a real problem here costs more than listing a wrong one.

**Static tools.** ruff, mypy, bandit and semgrep run on each PR's changed Python files. They run inside the PR's prepared image from the differential runner, at its last commit, network off. mypy runs with the environment's own Python and installed packages, so it sees the real types of the libraries the code uses.
- The tools live in their own image (`docker/tools/Dockerfile`), pinned by version and SHA-256. A copy of `/opt/tools` sits in a Docker volume that each static sandbox attaches read-only, so the PR's image stays unchanged.
- The rules are the same for every repo and ignore the repo's own settings:
  - ruff: pyflakes, bugbear and pylint's errors (`F,B,PLE`);
  - mypy: its defaults plus `--check-untyped-defs`, since most of these repos don't annotate their functions;
  - bandit: medium severity and up;
  - semgrep: the Python correctness and security rules of [semgrep-rules](https://github.com/semgrep/semgrep-rules) at a fixed commit (Semgrep Rules License).
- Warnings on lines the PR added or changed go to the `static_warnings` table (migration 0010). Every warning on the changed files stays in a JSON file under `data/static/`.

**The reviewer.** It is a LangGraph graph with one lookup round:
1. **The request.** It holds the PR's text (fitted as before), then the static warnings on changed lines as hints, then the retrieved context in the room left. Each part sits in its own tagged block, and tags inside any part are swapped for look-alikes.
2. **First reply.** The model replies with up to 15 candidates, or asks once for up to 4 lookups in the repo at the PR's last commit:
   - `read_file`: up to 60 lines of a file;
   - `search_symbol`: where a name is defined and called.
   The model only proposes lookups as JSON. Our code checks each one: a path must stay inside the repo, a name must be a Python name. It then answers them in an indexer sandbox with `ast`, network off (`context/lookup_tool.py`).
3. **Final reply.** The answers go into a final request, and the model must then reply with candidates.

One round keeps a PR at two calls at most, since each request nearly fills Groq's 8,000 tokens a minute.

```
pr-reviewer static run                    # the static tools on the 68 dev PRs from 2022 on (Docker; prepares PR images first)
pr-reviewer review candidates --with-issues --variant v2 --name candidates-gpt-oss-120b-v2
                                          # tune on the 26 PRs with known issues (recall depends only on them)
pr-reviewer score results/candidates-gpt-oss-120b-v2/raw/reviews.jsonl --name candidates-gpt-oss-120b-v2 --reviewer reviewer-gpt-oss-120b --since 2022 --with-issues
pr-reviewer review candidates --variant v1 # the final run on all 68, as candidates-gpt-oss-120b
pr-reviewer review compare --base context-gpt-oss-120b --new candidates-gpt-oss-120b
```

`results/static-dev-2022/` has the static summary: each tool's runs, warnings on the changed files and on changed lines, and rules.

**Tuning.** Two prompt variants ran on the 26 dev PRs with known issues. v1 asks for every problem that might be real. v2 also lists what to look for in each kind of problem, rules out an empty list, and drops a malformed candidate instead of rejecting the whole reply. Both found 12 of the 57 known issues, but v2 raised 107 false alarms against v1's 71, so the final run uses v1.

Results on the 68 dev PRs from 2022 onward:

| Reviewer | PR accuracy | Point precision | Point recall | Point F1 | Recall (F) | Recall (E) |
|---|---|---|---|---|---|---|
| No comments | 0.618 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Naive | 0.529 | 0.048 | 0.140 | 0.072 | 0.207 | 0.074 |
| Context | 0.485 | 0.076 | 0.175 | 0.106 | 0.276 | 0.074 |
| Candidates | 0.529 | 0.059 | 0.210 | 0.092 | 0.379 | 0.036 |

Recall rises from 10 to 12 hits, mostly on functional issues (8 → 11); precision falls, as expected for this stage. The counts are small, so one hit moves recall by 0.018.
- **Where it gains:** on the 32 PRs that got no retrieved context, hits go from 1 to 5. On the 36 with context, they go from 9 to 7.
- **Static warnings** landed on changed lines in only 7 PRs, two of them with known issues: sphinx-12491 got no hit, and sympy-26233 already had its hit in the context run, which showed no warnings. On sympy-24563 (clean) and sympy-26233, most candidates sat on warned lines, so the warnings mostly added false alarms.
- **Lookups** were asked on 11 PRs (1 hit there).
- **Clean PRs:** 123 false alarms on the 42 clean PRs (80 with context); 14 clean PRs got no candidate.

A full run uses about 355,000 tokens, close to two days of Groq's free quota. `results/candidates-gpt-oss-120b/` has the summaries and `comparison.md`; the tuning runs are in `results/candidates-gpt-oss-120b-v1/` and `-v2/`.

## Evidence agent

The evidence stage checks each candidate and records the strongest evidence it can get. The tiers, strongest first:
1. **Executed proof.** A generated test passes at the base and fails at the PR with the claimed error.
2. **Static-tool evidence.** A static warning on the claimed lines that the verifier confirms describes the same problem.
3. **Verified reasoning.** A verifier told to refute the claim fails to, and traces it: an input, the lines it runs and the failure.
4. **No evidence.** The gate, the next stage, drops these.

It runs as a LangGraph graph per PR:
1. **Survey.** One sandbox from the PR's prepared image reads what every candidate needs, with no model call: the function holding the claimed lines, the diff's nearest hunks, the existing test closest to the code (the one the retrieval step found using the function, or the nearest by folder), its imports and one of its tests, and the fixtures its folder's `conftest.py` files define. Files are parsed with `ast` on this side, so no repo code runs.
2. **One subgraph per candidate.** LangGraph's `Send` starts the candidate subgraph once per candidate, 2 at a time. Model calls go through one lock, one at a time, while another candidate's tests run in Docker.

Each candidate's subgraph:
- **Route.** A functional claim that names an error, in a Python file, goes to the test writer. Anything else, such as a documentation or design claim, goes straight to the verifier.
- **Write.** The model writes one pytest test of what the code should do, so it passes at the base and fails at the PR if the claim is true. Our code refuses a test that doesn't hold exactly one test function, or that uses calls that could fake where an error comes from (`exec`, `eval`, `compile`, swapping a function's code).
- **Run.** The differential runner runs it 3 times at each side, network off, watching the PR's changed source lines.
- **Match.** Our code reads the outcome:
  - **Proof:** the claimed error (or a subclass) at the PR, through a changed line, not raised by the test's own code.
  - **Wrong value:** for an `AssertionError` claim, the test's assert fails at the PR after running a changed line.
  - **Error content:** when the claim is about what an error says, a failed check on that error caught with `pytest.raises` also counts.
  - **New-code proof:** for code the PR adds, the base may fail only because a name the PR defines is missing.
  - **Broken test:** skipped, failing to load or set up, or failing at the base for another reason. It goes back to the model with the error, for a fix.
- **Intent check.** A proof is dropped when the PR's own tests, docs or changelog show the change was meant. Our code looks for signs in the diff; only when it finds some does a verifier call decide.
- **Verify.** The model tries to refute the claim and judges the static warnings on its lines.

**Budget per candidate:**
- **Test writer:** at most 3 replies (a test, then fixes or one round of lookup tool calls) and 20 minutes of test runs.
- **Verifier:** one decision, plus one optional lookup round.

**Lookups are tool calls.** The test writer and the verifier look things up by calling two tools, `read_file` and `search_symbol`, instead of listing lookups in their JSON reply as the candidate reviewer does. gpt-oss is trained for tool calls: asked to write lookups in JSON, it often "called" one in its thinking and ended its turn with no JSON, so Groq refused the reply, and at temperature 0 it did so again on every retry. Groq takes tools or a strict JSON format, not both, so a request that offers the tools has its JSON reply checked by our code alone; the request after the lookups offers no tools and uses the strict format.

Replies that aren't valid JSON are asked for again, as before, and so are replies Groq refuses (no JSON, broken JSON, or a call to a tool that doesn't exist). A refused call to a tool that doesn't exist is still read: a lookup under another name (such as `repo_browser.read_file`) counts as a lookup, and an answer sent as a tool's arguments counts as the reply if it passes the checks. After any other refused reply, or an empty one, the next tries send the same request without the tools. The evidence run uses the reviewer's model with room for 4,000 reply tokens (`evidence-gpt-oss-120b`), since at 3,000 about half the test writer's replies ran out while thinking.

Each candidate's tier, cost and evidence go to the `candidate_evidence` table (migration 0011), and its generated tests under `data/evidence/`. The run sleeps through Groq's quota waits (up to 30 minutes each) instead of stopping, so a retry in progress isn't lost. Model replies and test runs are cached, so a run that does stop resumes where it stopped.

```
pr-reviewer evidence run --with-issues    # the candidates of the 26 dev PRs with known issues
pr-reviewer evidence run --clean          # then those of the 42 clean ones
pr-reviewer evidence report               # write results/evidence-tools-gpt-oss-120b/ from what's saved
```

`evidence report` writes the tiers by kind, the proof rate, how the test writer and the verifier ended, and each tier set against the judge's verdict on the same finding in the candidate run, with no new judge call. Each proving test is copied to `proofs/`.

**Results.** These are for the 204 candidates of the candidate run, on 50 of the 68 dev PRs from 2022 onward. Each candidate is counted against the judge's verdict on the same finding:

| Tier | Candidates | Hit (known issue) | False alarm, PR with issues | False alarm, clean PR |
|---|---|---|---|---|
| 1 (executed) | 5 | 2 | 3 | 0 |
| 2 (static tool) | 4 | 0 | 0 | 4 |
| 3 (verifier) | 20 | 0 | 7 | 13 |
| 4 (dropped) | 175 | 10 | 59 | 106 |

- **The evidence removes most false alarms, and most hits too.** Tiers 1–3 keep 29 of the 204 candidates. That is 2 of the 12 hits and 27 of the 192 false alarms; 17 of those 27 are on clean PRs, against 123 before.
- **The verifier is the weak link:**
  - It refuted all 10 hits it judged.
  - None of its 20 upheld claims is a hit.
  - It never sees the test runs, or whether a claimed line is new, moved or unchanged. Some upheld claims were already disproved by a test run. Others describe a change the PR's own tests show was meant, or a problem older than the PR.
- **Executed proof is rare:**
  - 5 of the 148 functional candidates got a proving test (3.4%), and 5 of the 115 that name an error (4.3%). No clean PR got one.
  - Of the 83 candidates sent to the test writer, the model wrote 157 tests. Our checks refused 51 before running; 43 of those caught the claimed error with `pytest.raises`, so they would pass where the bug is. Another 61 broke when run, 46 of them by failing at the base commit.
  - A first look at the 5 proofs found weak spots in 4. In 3, the base passes only through a guard the test adds around code the PR introduces. In 2, the PR's own tests or docs show the change was meant, and the intent check found no signs, so it never ran. (One proof has both problems.)
- **Groq refused about a quarter of the calls** (137 of 579). 104 of those held a usable answer sent as a call to a made-up tool, and were read back.
- **Cost:**
  - 574 model replies, using about 865,000 tokens in and 295,000 out. Read-back answers are logged at 0 tokens, but Groq still counts them.
  - 84 minutes of test runs in Docker.
  - About 6.5 days on the free quota.

`results/evidence-tools-gpt-oss-120b/` has the summary, the config and the 5 proving tests.

## Sandbox (Docker)

PR code is untrusted, so it only ever runs inside Docker containers called sandboxes. Every sandbox:

- has the limits set in `configs/sandbox.yaml`: 2 CPUs, 2 GB of memory with no swap, 512 processes, and a 2-hour lifetime, after which it stops itself even if the program that started it crashed. Each command also has a time limit: 30 minutes for installs, 15 for tests.
- runs commands as a non-root user with no Linux privileges. It never sees your files, your `.env` settings or Docker itself.
- starts with its network off: it has nothing but its own loopback interface, so even name lookups fail.

When a step needs the network, such as installing packages, the sandbox joins an internal Docker network whose only way out is a proxy container. The proxy (tinyproxy, also locked down) forwards only to the hosts listed in `configs/sandbox.yaml`: GitHub, PyPI, Debian, conda-forge and python.org. So install code can't reach other sites, or services on this machine such as the database and pgAdmin. To allow another host, add it to `allowed_hosts`; the proxy restarts with the new list.

The SWR-Bench repos are cloned once, inside Docker, into the Docker volume `pr-reviewer-repos`, never onto the Windows drive. Each PR's commits come from GitHub's `pull/<N>/head` refs, as SWR-Bench's own scripts get them. Sandboxes see these copies read-only and make their own working copy from one in seconds, so checkouts work with the network off.

A PR has two sides:
- **base**: the commit just before its first reviewed commit, which is what SWR-Bench's own scripts use. The dataset's `base_commit` field can be a different, later commit.
- **PR state**: its last reviewed commit, the code the reviewer sees.

```
pr-reviewer sandbox build      # build the sandbox and proxy images (first time: about 11 minutes)
pr-reviewer sandbox repos      # download the dev PRs' repos and commits (about 1.6 GB, once)
pr-reviewer sandbox check      # check every dev PR's commits and both checkouts, with the network off
pr-reviewer sandbox run <id> --at base -- git log -1    # run a command at a dev PR's base (or --at pr)
pr-reviewer sandbox status     # images, proxy, repo copies and their sizes, leftover containers
pr-reviewer sandbox cleanup    # remove sandbox containers left behind by a crash
```

`sandbox run` takes `--network` to turn the network on (through the proxy), and refuses test-set PRs. `sandbox check` writes `results/commit-check/` and the `pr_checkouts` table. For each dev PR, these record its base and PR state, whether both can be checked out, how the `base_commit` field relates to the base, and any merge commits. Each sandbox command's output is logged under `data/sandbox-logs/`.

On the dev set, all 130 PRs can be checked out at both sides, with the network off. For 75 of them the `base_commit` field is the base itself. For the other 55 it's later on the main line: 7 commits after the base in the middle case, and up to 3,317. Testing from that field would mix other people's changes into the PR's, which is why the base is taken from git instead.

## Environment setup (the setup agent)

To test a PR, its repo must be installed as it was at the time. The setup agent builds that environment for each dev PR. Only the repo's own tests decide whether it works, never the agent.

1. It reads the repo's setup files at the PR's base, such as `setup.py`, `pyproject.toml`, the CI configs and the requirement files.
2. It picks the existing tests related to the PR: test files the PR edits, or ones named after the modules it changes, at most 5 files.
3. It tries, in order:
   - a recipe that worked for an earlier PR of the same repo and year, or the nearest year
   - the standard recipe: the right Python, the repo in editable mode with its test extra and requirements, and pytest
   - recipes from the setup agent's model (`setup-gpt-oss-120b`), which reads the repo's facts and each failed attempt's log, and may ask to read files first

   The budget is 12 model calls and 60 minutes per PR. A setup that runs out gives up and records why, in a category named from the logs.
4. Each attempt runs in a fresh sandbox:
   - **Base:** install with the network on (through the proxy), then run the related tests with the network off.
   - **PR state:** switch to it, reinstall, and run the same tests again.

   The environment is verified when at least 90% of those tests pass both times. It's then saved as a Docker image (`pr-reviewer-env:...`), so later PRs with the same recipe start from it.

Packages are pinned to the PR's date (`UV_EXCLUDE_NEWER`), so an old commit gets the libraries of its time. The one exception: for PRs before September 2022, setuptools may be as new as September 2022, the first versions that can do editable installs. uv installs Python 3.8 and newer; Python 3.7, for 2019 and 2020 PRs, is built into the sandbox image from python.org's source. The image also holds a fixed set of compilers and libraries; recipes can't install system packages.

Progress is saved in PostgreSQL after each step, so a setup stopped by the model's daily limit continues where it stopped. The free tier allows about 35 model calls a day.

```
pr-reviewer setup run                  # set up the dev PRs from 2022 onward not set up yet
pr-reviewer setup run --since 2019     # also the older ones
pr-reviewer setup run --ids <id> ...   # only these PRs (--again to start them over)
pr-reviewer setup report               # write results/setup-dev/ for the dev PRs from 2022 onward
pr-reviewer setup report --since 2019  # the same, with the older ones
```

`setup report` writes `results/setup-dev/`:
- `summary.md`: success rates by repo and year, what got environments working, and why setups failed
- `usable_prs.json`: the fixed usable subset, the verified dev PRs that later experiments run on
- `config.json`: how the setups ran

Recipes are kept in the `env_recipes` table, and every setup in `setup_runs`.

On the dev set, all 68 PRs from 2022 onward are verified, in 9 repos: 51 with a recipe kept from an earlier PR, 14 with the standard recipe, and 3 with the model's recipes (14 model calls). The 62 dev PRs from 2019–2021 aren't set up yet.

## Differential test runner

The strongest evidence for a finding is a test that passes before the PR and fails after it. The differential runner runs the same tests at a PR's base and at its PR state and compares, test by test:

1. Each set-up PR is installed once at its base and once at its PR state, and both are saved as Docker images (`pr-reviewer-pr:<id>-base` and `-pr`), so later runs start in seconds.
2. The test files are written into the working copy at both sides, so both run the same test code and only the code under test differs.
3. Each side's tests run 3 times, with the network off and at most 5 minutes a run. A small pytest plugin, loaded from outside the repo, records each test's result: its exception type and parent classes, message, failing line and traceback. Given lines to watch (such as the PR's changed lines), it also records which of them each test ran, with Python's line tracer. Django's runner output is read instead.
4. Each test gets one outcome:
   - **pass→fail:** the PR broke what the test checks. Only this outcome can prove a bug.
   - **pass→pass, fail→pass, fail→fail:** the other three.
   - **flaky:** the runs at one side disagreed.
   - **not run:** a timeout, out of memory, skipped, or never collected.

   Flaky and not-run tests never count as proof, so a timeout can't pass for a failure.
5. A test method defined in a base class or mixin runs under every test class in the file that inherits it, since pytest never runs it under the base's own name. If pytest stops because a requested test sits in a file that didn't load (a broken import, or `pytest.importorskip()` at the top), those tests count as errors or skips, and the others run again without them.

Each run is stored in the `diff_runs` table under a fingerprint of the images, the test code and the settings. The same test on the same environment is never run twice.

```
pr-reviewer difftest prepare [--ids <id> ...]          # save each set-up dev PR's base and PR-state images
pr-reviewer difftest run <id> <test> [<test> ...]     # run tests (node IDs, files or Django labels) at both sides
pr-reviewer difftest run <id> --file t.py --path pkg/tests/test_new.py   # a test file of your own
pr-reviewer difftest validate                         # check the runner on known tests
```

`difftest validate` checks the runner on dev PRs whose tests have a known outcome:
- a PR's own new or edited tests should go fail→pass or pass→pass, never pass→fail;
- a known bug's fix commit that adds a test should make that test fail at the PR state and pass at the fix;
- one PR's related tests in each other repo should stay as they were.

Only the tests the PR or fix added or changed run, found from the diff. The results go to `results/difftest-validation/`. Tests that the repo's own skip marks skip at both sides (such as a missing optional package) are counted apart, as neither expected nor not.

On 28 set-up dev PRs (36 cases, 1,174 tests), from the base to the PR state:

| Kind | Cases | Tests | pass→fail | flaky | not run | skipped | As expected |
|---|---|---|---|---|---|---|---|
| PR's own tests | 23 | 857 | 0 | 0 | 0 | 473 | 382 of 384 |
| related tests | 4 | 141 | 0 | 0 | 0 | 14 | 127 of 127 |

- No false proof: no test went pass→fail, and none was flaky. The 2 unexpected tests (xarray) need dask, which the environment lacks, so they fail the same way at both sides.
- **Known bugs:** the fixes' tests caught 4 of 9 known bugs (they fail at the PR state and pass at the fix). The other 5 can't show here:
  - 2 are mistakes in the PR's own test code;
  - 1 shows only on machines without a C compiler;
  - 1 needs dask;
  - 1 fix raised numpy's minimum version, so it doesn't install with the PR's recipe.
- None of the caught bugs' tests pass at the base, so none shows as pass→fail. They use what the PR itself added, such as a new keyword argument, which the base lacks.
- The skipped tests are mostly xarray's, which need optional packages (dask and others) that the environments don't install.
- The 58 prepared images (29 PRs × 2 sides) add about 6.3 GB on top of the environment images they share layers with.

## Checks

```
pytest                        # integration and docker tests are skipped when the database or Docker isn't running
pytest -m "not integration and not docker"   # unit tests only
pytest -m docker              # the sandbox tests (need the images built; a few need internet)
ruff check .
mypy
```
