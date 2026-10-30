# Scores: baseline-known-issues

- Reviews: `known-issues.jsonl`, 130 PRs of the dev set (39 with known issues, 91 clean)
- Reviewer: none
- Judge: gemini-3.5-flash-lite (temperature 0.0, replies up to 8,192 tokens), SWR-Bench's scorer at 67ae1d4
- Scored: 2026-10-30

## PR level: does the review flag PRs with issues and pass clean ones?

| Accuracy | Precision | Recall | F1 | Flagged, with issues | Missed, with issues | False alarm, clean | Passed, clean |
|---|---|---|---|---|---|---|---|
| 1.000 | 1.000 | 1.000 | 1.000 | 39 | 0 | 0 | 91 |

A review passes a PR when the judge reads it as approving, or finds no point in it.

## Point level: do the review's points match the known issues?

| Issues | Precision | Recall | F1 | Hits | False alarms | Misses |
|---|---|---|---|---|---|---|
| All | 1.000 | 1.000 | 1.000 | 82 | 0 | 0 |
| Functional (F) | 1.000 | 1.000 | 1.000 | 44 | 0 | 0 |
| Evolvability (E) | 1.000 | 1.000 | 1.000 | 38 | 0 | 0 |

A point is a hit when it matches a known issue and a false alarm when it matches none (every point on a clean PR is one). A known issue no point matches is a miss. One point matching several issues counts as one hit, the rest as misses (SWR-Bench's rule). The F and E rows sort points by the category the judge gave them.

Review points per PR: 0.63.
