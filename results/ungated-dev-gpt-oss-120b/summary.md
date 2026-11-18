# Scores: ungated-dev-gpt-oss-120b

- Reviews: `reviews.jsonl`, 68 PRs of the dev set from 2022 on (26 with known issues, 42 clean)
- Reviewer: openai/gpt-oss-120b
- Judge: gemini-3.5-flash-lite (temperature 0.0, replies up to 8,192 tokens), SWR-Bench's scorer at 67ae1d4
- Scored: 2026-11-17

## PR level: does the review flag PRs with issues and pass clean ones?

| Accuracy | Precision | Recall | F1 | Flagged, with issues | Missed, with issues | False alarm, clean | Passed, clean |
|---|---|---|---|---|---|---|---|
| 0.529 | 0.440 | 0.846 | 0.579 | 22 | 4 | 28 | 14 |

A review passes a PR when the judge reads it as approving, or finds no point in it.

## Point level: do the review's points match the known issues?

| Issues | Precision | Recall | F1 | Hits | False alarms | Misses |
|---|---|---|---|---|---|---|
| All | 0.059 | 0.210 | 0.092 | 12 | 192 | 45 |
| Functional (F) | 0.068 | 0.379 | 0.116 | 11 | 150 | 18 |
| Evolvability (E) | 0.023 | 0.036 | 0.028 | 1 | 42 | 27 |

A point is a hit when it matches a known issue and a false alarm when it matches none (every point on a clean PR is one). A known issue no point matches is a miss. One point matching several issues counts as one hit, the rest as misses (SWR-Bench's rule). The F and E rows sort points by the category the judge gave them.

Review points per PR: 3.00.
