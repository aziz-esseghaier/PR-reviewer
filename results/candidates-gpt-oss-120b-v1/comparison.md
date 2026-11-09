# candidates-gpt-oss-120b-v1 compared with context-gpt-oss-120b

The same 26 PRs, scored by the same judge. Metrics on a subset use SWR-Bench's formulas over that subset's judgments (no new judge call); on all PRs they match SWR-Bench's own numbers.

| PRs | Run | Accuracy | Precision | Recall | F1 | Recall (F) | F1 (F) | Recall (E) | F1 (E) | Hits | False alarms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| All 26 | context-gpt-oss-120b | 0.692 | 0.192 | 0.175 | 0.183 | 0.276 | 0.235 | 0.074 | 0.100 | 10 | 42 |
| All 26 | candidates-gpt-oss-120b-v1 | 0.846 | 0.145 | 0.210 | 0.171 | 0.379 | 0.250 | 0.036 | 0.038 | 12 | 71 |
| 17 with context | context-gpt-oss-120b | 0.824 | 0.220 | 0.220 | 0.220 | 0.368 | 0.264 | 0.095 | 0.143 | 9 | 32 |
| 17 with context | candidates-gpt-oss-120b-v1 | 0.882 | 0.149 | 0.171 | 0.159 | 0.316 | 0.214 | 0.045 | 0.062 | 7 | 40 |
| 9 without context | context-gpt-oss-120b | 0.444 | 0.091 | 0.062 | 0.074 | 0.100 | 0.133 | 0.000 | 0.000 | 1 | 10 |
| 9 without context | candidates-gpt-oss-120b-v1 | 0.778 | 0.139 | 0.312 | 0.192 | 0.500 | 0.312 | 0.000 | 0.000 | 5 | 31 |
| 3 with lookups | context-gpt-oss-120b | 1.000 | 0.111 | 0.200 | 0.143 | 0.333 | 0.167 | 0.000 | 0.000 | 1 | 8 |
| 3 with lookups | candidates-gpt-oss-120b-v1 | 0.667 | 0.077 | 0.200 | 0.111 | 0.333 | 0.143 | 0.000 | 0.000 | 1 | 12 |
| 23 without lookups | context-gpt-oss-120b | 0.652 | 0.209 | 0.173 | 0.190 | 0.269 | 0.250 | 0.080 | 0.105 | 9 | 34 |
| 23 without lookups | candidates-gpt-oss-120b-v1 | 0.870 | 0.157 | 0.211 | 0.180 | 0.385 | 0.270 | 0.038 | 0.042 | 11 | 59 |

## PR by PR

- Hits: 12 with candidates-gpt-oss-120b-v1, 10 with context-gpt-oss-120b; 5 PRs gained hits, 2 lost some
- False alarms on the 0 clean PRs: 0 (was 0); clean PRs passed: 0 (was 0)
- False alarms on the 26 PRs with issues: 71 (was 42); PRs with issues flagged: 22 (was 18)

| PR | Issues | Hits (context-gpt-oss-120b → candidates-gpt-oss-120b-v1) | False alarms |
|---|---|---|---|
| astropy__astropy-12747 | 1 | 0 → 1 | 1 → 4 |
| astropy__astropy-16735 | 3 | 0 → 1 | 0 → 4 |
| astropy__astropy-17268 | 2 | 0 → 0 | 2 → 1 |
| matplotlib__matplotlib-24674 | 1 | 1 → 1 | 2 → 4 |
| matplotlib__matplotlib-27750 | 1 | 0 → 1 | 0 → 7 |
| pydata__xarray-6260 | 2 | 1 → 1 | 1 → 2 |
| pydata__xarray-6939 | 1 | 1 → 1 | 1 → 0 |
| pydata__xarray-8246 | 3 | 2 → 0 | 8 → 5 |
| pylint-dev__pylint-7257 | 1 | 0 → 0 | 1 → 2 |
| pylint-dev__pylint-7918 | 1 | 0 → 0 | 0 → 7 |
| pylint-dev__pylint-8706 | 3 | 0 → 0 | 6 → 5 |
| pytest-dev__pytest-12232 | 3 | 1 → 1 | 0 → 5 |
| scikit-learn__scikit-learn-22957 | 2 | 0 → 0 | 2 → 1 |
| scikit-learn__scikit-learn-23459 | 2 | 0 → 0 | 2 → 1 |
| scikit-learn__scikit-learn-25108 | 4 | 1 → 0 | 2 → 2 |
| scikit-learn__scikit-learn-25148 | 9 | 1 → 1 | 2 → 3 |
| scikit-learn__scikit-learn-26928 | 2 | 0 → 1 | 1 → 3 |
| scikit-learn__scikit-learn-27943 | 1 | 0 → 0 | 0 → 2 |
| scikit-learn__scikit-learn-28564 | 2 | 0 → 1 | 0 → 0 |
| scikit-learn__scikit-learn-30581 | 1 | 1 → 1 | 2 → 1 |
| sphinx-doc__sphinx-12491 | 1 | 0 → 0 | 5 → 8 |
| sympy__sympy-25735 | 3 | 0 → 0 | 1 → 0 |
| sympy__sympy-26233 | 3 | 1 → 1 | 3 → 4 |
