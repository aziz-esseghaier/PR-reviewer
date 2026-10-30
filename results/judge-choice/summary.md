# Judge candidates on dev PRs with known right answers

| Model | As expected | Valid JSON 1st try | Known issues hit | Hits on wrong reviews | "No issues" read right | Same answer twice | Median s/call | Tokens in/out |
|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | 24/24 | 24/24 | 16/16 | 0 | 7/7 | 5/5 | 6.0 | 156,919 / 13,708 |
| gemma-4-26b-a4b-it | 24/24 | 23/24 | 16/16 | 0 | 7/7 | 5/5 | 9.6 | 156,919 / 12,602 |
| gemini-3.5-flash-lite | 24/24 | 24/24 | 16/16 | 0 | 7/7 | 5/5 | 5.8 | 156,919 / 12,963 |

## Judgments that weren't as expected

None.
