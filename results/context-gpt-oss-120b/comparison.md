# context-gpt-oss-120b compared with naive-gpt-oss-120b

The same 68 PRs, scored by the same judge. Metrics on a subset use SWR-Bench's formulas over that subset's judgments (no new judge call); on all PRs they match SWR-Bench's own numbers.

| PRs | Run | Accuracy | Precision | Recall | F1 | F1 (F) | F1 (E) | Hits | False alarms |
|---|---|---|---|---|---|---|---|---|---|
| All 68 | naive-gpt-oss-120b | 0.529 | 0.048 | 0.140 | 0.072 | 0.087 | 0.048 | 8 | 158 |
| All 68 | context-gpt-oss-120b | 0.485 | 0.076 | 0.175 | 0.106 | 0.136 | 0.057 | 10 | 122 |
| 37 with context | naive-gpt-oss-120b | 0.595 | 0.051 | 0.125 | 0.073 | 0.096 | 0.037 | 5 | 92 |
| 37 with context | context-gpt-oss-120b | 0.514 | 0.111 | 0.175 | 0.136 | 0.191 | 0.050 | 7 | 56 |
| 31 without context | naive-gpt-oss-120b | 0.452 | 0.043 | 0.176 | 0.070 | 0.073 | 0.067 | 3 | 66 |
| 31 without context | context-gpt-oss-120b | 0.452 | 0.043 | 0.176 | 0.070 | 0.073 | 0.067 | 3 | 66 |

## PR by PR

- Hits: 10 with context-gpt-oss-120b, 8 with naive-gpt-oss-120b; 4 PRs gained hits, 2 lost some
- False alarms on the 42 clean PRs: 80 (was 92); clean PRs passed: 15 (was 15)
- False alarms on the 26 PRs with issues: 42 (was 66); PRs with issues flagged: 18 (was 21)

| PR | Issues | Hits (naive-gpt-oss-120b → context-gpt-oss-120b) | False alarms |
|---|---|---|---|
| astropy__astropy-16735 | 3 | 1 → 0 | 4 → 0 |
| astropy__astropy-17268 | 2 | 0 → 0 | 4 → 2 |
| astropy__astropy-17800 | 0 | 0 → 0 | 3 → 0 |
| matplotlib__matplotlib-24674 | 1 | 0 → 1 | 5 → 2 |
| matplotlib__matplotlib-28677 | 0 | 0 → 0 | 1 → 2 |
| pydata__xarray-10006 | 0 | 0 → 0 | 0 → 1 |
| pydata__xarray-6260 | 2 | 1 → 1 | 2 → 1 |
| pydata__xarray-6710 | 0 | 0 → 0 | 3 → 5 |
| pydata__xarray-6939 | 1 | 1 → 1 | 4 → 1 |
| pytest-dev__pytest-12232 | 3 | 0 → 1 | 3 → 0 |
| scikit-learn__scikit-learn-23459 | 2 | 0 → 0 | 5 → 2 |
| scikit-learn__scikit-learn-24456 | 0 | 0 → 0 | 3 → 1 |
| scikit-learn__scikit-learn-25108 | 4 | 0 → 1 | 3 → 2 |
| scikit-learn__scikit-learn-25148 | 9 | 1 → 1 | 3 → 2 |
| scikit-learn__scikit-learn-27943 | 1 | 0 → 0 | 3 → 0 |
| scikit-learn__scikit-learn-28564 | 2 | 1 → 0 | 0 → 0 |
| scikit-learn__scikit-learn-31106 | 0 | 0 → 0 | 10 → 4 |
| sympy__sympy-22957 | 0 | 0 → 0 | 0 → 1 |
| sympy__sympy-22973 | 0 | 0 → 0 | 2 → 4 |
| sympy__sympy-23450 | 0 | 0 → 0 | 5 → 4 |
| sympy__sympy-24830 | 0 | 0 → 0 | 8 → 2 |
| sympy__sympy-26233 | 3 | 0 → 1 | 3 → 3 |
| sympy__sympy-26660 | 0 | 0 → 0 | 1 → 0 |
