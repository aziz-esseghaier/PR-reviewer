# Scores: baseline-known-issues-2022

- Reviews: `known-issues.jsonl`, 68 PRs of the dev set from 2022 on (26 with known issues, 42 clean)
- Reviewer: none
- Judge: gemini-3.5-flash-lite (temperature 0.0, replies up to 8,192 tokens), SWR-Bench's scorer at 67ae1d4
- Scored: 2026-11-05

## PR level: does the review flag PRs with issues and pass clean ones?

| Accuracy | Precision | Recall | F1 | Flagged, with issues | Missed, with issues | False alarm, clean | Passed, clean |
|---|---|---|---|---|---|---|---|
| 1.000 | 1.000 | 1.000 | 1.000 | 26 | 0 | 0 | 42 |

A review passes a PR when the judge reads it as approving, or finds no point in it.

## Point level: do the review's points match the known issues?

| Issues | Precision | Recall | F1 | Hits | False alarms | Misses |
|---|---|---|---|---|---|---|
| All | 1.000 | 1.000 | 1.000 | 57 | 0 | 0 |
| Functional (F) | 1.000 | 1.000 | 1.000 | 29 | 0 | 0 |
| Evolvability (E) | 1.000 | 1.000 | 1.000 | 28 | 0 | 0 |

A point is a hit when it matches a known issue and a false alarm when it matches none (every point on a clean PR is one). A known issue no point matches is a miss. One point matching several issues counts as one hit, the rest as misses (SWR-Bench's rule). The F and E rows sort points by the category the judge gave them.

Review points per PR: 0.84.
