"""Measures the reviewer on SWR-Bench.

- source.py: where the SWR-Bench files come from (one pinned commit), with downloading and
  fingerprint checks.
- dataset.py: reads the PRs, checks each record, and picks the ones the benchmark uses.

Still to come: the dev/test split, scoring with SWR-Bench's official evaluation script, and
the analysis of results.

Imported, not run.
"""
