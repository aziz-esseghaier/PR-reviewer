"""Measures the reviewer on SWR-Bench.

- source.py: where the SWR-Bench files come from (one pinned commit), with downloading and
  fingerprint checks.
- dataset.py: reads the PRs, checks each record (including that no fix is reviewed), and
  picks the ones the benchmark uses.
- review_input.py: the text a reviewer sees for a PR, written as SWR-Bench writes it.
- split.py: the fixed dev/test split, and the lock that keeps the test set unused.
- official.py: loads SWR-Bench's scoring script unchanged, with our run_chat in place of its own.
- judge.py: the fixed judge model, and the run_chat that sends the scorer's calls through our
  LLM client.
- scoring.py: scores one review per PR with SWR-Bench's scorer (`pr-reviewer score`).
- results.py: keeps the scores, in results/<name>/ and in PostgreSQL.
- baselines.py: reviews written without a model, to give the score to beat and check the
  scoring.

Still to come: the analysis of results across experiments.

Imported, not run.
"""
