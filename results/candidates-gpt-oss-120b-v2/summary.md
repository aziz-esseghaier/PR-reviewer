# Scores: candidates-gpt-oss-120b-v2

- Reviews: `reviews.jsonl`, 26 PRs of the dev set from 2022 on, only those with known issues (26 with known issues, 0 clean)
- Reviewer: openai/gpt-oss-120b
- Judge: gemini-3.5-flash-lite (temperature 0.0, replies up to 8,192 tokens), SWR-Bench's scorer at 67ae1d4
- Scored: 2026-11-08

## PR level: does the review flag PRs with issues and pass clean ones?

| Accuracy | Precision | Recall | F1 | Flagged, with issues | Missed, with issues | False alarm, clean | Passed, clean |
|---|---|---|---|---|---|---|---|
| 1.000 | 1.000 | 1.000 | 1.000 | 26 | 0 | 0 | 0 |

A review passes a PR when the judge reads it as approving, or finds no point in it.

## Point level: do the review's points match the known issues?

| Issues | Precision | Recall | F1 | Hits | False alarms | Misses |
|---|---|---|---|---|---|---|
| All | 0.101 | 0.210 | 0.136 | 12 | 107 | 45 |
| Functional (F) | 0.154 | 0.333 | 0.210 | 10 | 55 | 20 |
| Evolvability (E) | 0.037 | 0.074 | 0.049 | 2 | 52 | 25 |

A point is a hit when it matches a known issue and a false alarm when it matches none (every point on a clean PR is one). A known issue no point matches is a miss. One point matching several issues counts as one hit, the rest as misses (SWR-Bench's rule). The F and E rows sort points by the category the judge gave them.

Review points per PR: 4.58.
