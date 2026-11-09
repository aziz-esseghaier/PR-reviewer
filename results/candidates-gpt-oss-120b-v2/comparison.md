# candidates-gpt-oss-120b-v2 compared with context-gpt-oss-120b

The same 26 PRs, scored by the same judge. Metrics on a subset use SWR-Bench's formulas over that subset's judgments (no new judge call); on all PRs they match SWR-Bench's own numbers.

| PRs | Run | Accuracy | Precision | Recall | F1 | Recall (F) | F1 (F) | Recall (E) | F1 (E) | Hits | False alarms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| All 26 | context-gpt-oss-120b | 0.692 | 0.192 | 0.175 | 0.183 | 0.276 | 0.235 | 0.074 | 0.100 | 10 | 42 |
| All 26 | candidates-gpt-oss-120b-v2 | 1.000 | 0.101 | 0.210 | 0.136 | 0.333 | 0.210 | 0.074 | 0.049 | 12 | 107 |
| 14 with context | context-gpt-oss-120b | 0.786 | 0.261 | 0.214 | 0.235 | 0.375 | 0.353 | 0.000 | 0.000 | 6 | 17 |
| 14 with context | candidates-gpt-oss-120b-v2 | 1.000 | 0.115 | 0.214 | 0.150 | 0.294 | 0.217 | 0.091 | 0.059 | 6 | 46 |
| 12 without context | context-gpt-oss-120b | 0.583 | 0.138 | 0.138 | 0.138 | 0.154 | 0.118 | 0.133 | 0.174 | 4 | 25 |
| 12 without context | candidates-gpt-oss-120b-v2 | 1.000 | 0.090 | 0.207 | 0.125 | 0.385 | 0.204 | 0.062 | 0.043 | 6 | 61 |
| 7 with lookups | context-gpt-oss-120b | 0.714 | 0.286 | 0.191 | 0.229 | 0.429 | 0.353 | 0.071 | 0.111 | 4 | 10 |
| 7 with lookups | candidates-gpt-oss-120b-v2 | 1.000 | 0.083 | 0.143 | 0.105 | 0.143 | 0.080 | 0.143 | 0.125 | 3 | 33 |
| 19 without lookups | context-gpt-oss-120b | 0.684 | 0.158 | 0.167 | 0.162 | 0.227 | 0.196 | 0.077 | 0.091 | 6 | 32 |
| 19 without lookups | candidates-gpt-oss-120b-v2 | 1.000 | 0.108 | 0.250 | 0.151 | 0.391 | 0.257 | 0.000 | 0.000 | 9 | 74 |

## PR by PR

- Hits: 12 with candidates-gpt-oss-120b-v2, 10 with context-gpt-oss-120b; 7 PRs gained hits, 4 lost some
- False alarms on the 0 clean PRs: 0 (was 0); clean PRs passed: 0 (was 0)
- False alarms on the 26 PRs with issues: 107 (was 42); PRs with issues flagged: 26 (was 18)

| PR | Issues | Hits (context-gpt-oss-120b → candidates-gpt-oss-120b-v2) | False alarms |
|---|---|---|---|
| astropy__astropy-12747 | 1 | 0 → 1 | 1 → 7 |
| astropy__astropy-16735 | 3 | 0 → 1 | 0 → 6 |
| matplotlib__matplotlib-24674 | 1 | 1 → 0 | 2 → 4 |
| matplotlib__matplotlib-25737 | 2 | 0 → 0 | 0 → 3 |
| matplotlib__matplotlib-27750 | 1 | 0 → 1 | 0 → 6 |
| pydata__xarray-6260 | 2 | 1 → 0 | 1 → 2 |
| pydata__xarray-6939 | 1 | 1 → 1 | 1 → 9 |
| pydata__xarray-8246 | 3 | 2 → 0 | 8 → 7 |
| pylint-dev__pylint-7257 | 1 | 0 → 0 | 1 → 3 |
| pylint-dev__pylint-7918 | 1 | 0 → 0 | 0 → 4 |
| pylint-dev__pylint-8706 | 3 | 0 → 0 | 6 → 8 |
| pytest-dev__pytest-12232 | 3 | 1 → 1 | 0 → 1 |
| scikit-learn__scikit-learn-22454 | 1 | 0 → 1 | 0 → 1 |
| scikit-learn__scikit-learn-22729 | 2 | 0 → 0 | 0 → 3 |
| scikit-learn__scikit-learn-23459 | 2 | 0 → 0 | 2 → 5 |
| scikit-learn__scikit-learn-25148 | 9 | 1 → 1 | 2 → 6 |
| scikit-learn__scikit-learn-26928 | 2 | 0 → 1 | 1 → 2 |
| scikit-learn__scikit-learn-27943 | 1 | 0 → 0 | 0 → 6 |
| scikit-learn__scikit-learn-28564 | 2 | 0 → 1 | 0 → 0 |
| scikit-learn__scikit-learn-30581 | 1 | 1 → 1 | 2 → 4 |
| sympy__sympy-25735 | 3 | 0 → 1 | 1 → 4 |
| sympy__sympy-26233 | 3 | 1 → 0 | 3 → 5 |
