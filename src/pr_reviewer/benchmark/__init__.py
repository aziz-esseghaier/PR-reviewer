"""Measures the reviewer on SWR-Bench.

- source.py: where the SWR-Bench files come from (one pinned commit), with downloading and
  fingerprint checks.
- dataset.py: reads the PRs, checks each record (including that no fix is reviewed), and
  picks the ones the benchmark uses.
- review_input.py: the text a reviewer sees for a PR, written as SWR-Bench writes it.
- split.py: the fixed dev/test split, and the lock that keeps the test set unused.

Still to come: scoring with SWR-Bench's official evaluation script, and the analysis of
results.

Imported, not run.
"""
