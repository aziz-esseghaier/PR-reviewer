# candidates-gpt-oss-120b compared with context-gpt-oss-120b

The same 68 PRs, scored by the same judge. Metrics on a subset use SWR-Bench's formulas over that subset's judgments (no new judge call); on all PRs they match SWR-Bench's own numbers.

| PRs | Run | Accuracy | Precision | Recall | F1 | Recall (F) | F1 (F) | Recall (E) | F1 (E) | Hits | False alarms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| All 68 | context-gpt-oss-120b | 0.485 | 0.076 | 0.175 | 0.106 | 0.276 | 0.136 | 0.074 | 0.057 | 10 | 122 |
| All 68 | candidates-gpt-oss-120b | 0.529 | 0.059 | 0.210 | 0.092 | 0.379 | 0.116 | 0.036 | 0.028 | 12 | 192 |
| 36 with context | context-gpt-oss-120b | 0.583 | 0.125 | 0.220 | 0.159 | 0.368 | 0.187 | 0.095 | 0.108 | 9 | 63 |
| 36 with context | candidates-gpt-oss-120b | 0.556 | 0.080 | 0.171 | 0.108 | 0.316 | 0.128 | 0.045 | 0.057 | 7 | 81 |
| 32 without context | context-gpt-oss-120b | 0.375 | 0.017 | 0.062 | 0.026 | 0.100 | 0.046 | 0.000 | 0.000 | 1 | 59 |
| 32 without context | candidates-gpt-oss-120b | 0.500 | 0.043 | 0.312 | 0.076 | 0.500 | 0.104 | 0.000 | 0.000 | 5 | 111 |
| 11 with lookups | context-gpt-oss-120b | 0.455 | 0.045 | 0.250 | 0.077 | 0.500 | 0.100 | 0.000 | 0.000 | 1 | 21 |
| 11 with lookups | candidates-gpt-oss-120b | 0.273 | 0.033 | 0.250 | 0.059 | 0.500 | 0.091 | 0.000 | 0.000 | 1 | 29 |
| 57 without lookups | context-gpt-oss-120b | 0.491 | 0.082 | 0.170 | 0.110 | 0.259 | 0.143 | 0.080 | 0.062 | 9 | 101 |
| 57 without lookups | candidates-gpt-oss-120b | 0.579 | 0.063 | 0.207 | 0.097 | 0.370 | 0.119 | 0.038 | 0.034 | 11 | 163 |

## PR by PR

- Hits: 12 with candidates-gpt-oss-120b, 10 with context-gpt-oss-120b; 5 PRs gained hits, 2 lost some
- False alarms on the 42 clean PRs: 123 (was 80); clean PRs passed: 14 (was 15)
- False alarms on the 26 PRs with issues: 69 (was 42); PRs with issues flagged: 22 (was 18)

| PR | Issues | Hits (context-gpt-oss-120b → candidates-gpt-oss-120b) | False alarms |
|---|---|---|---|
| astropy__astropy-12747 | 1 | 0 → 1 | 1 → 4 |
| astropy__astropy-14854 | 0 | 0 → 0 | 10 → 0 |
| astropy__astropy-15549 | 0 | 0 → 0 | 0 → 3 |
| astropy__astropy-16539 | 0 | 0 → 0 | 2 → 5 |
| astropy__astropy-16735 | 3 | 0 → 1 | 0 → 4 |
| astropy__astropy-17268 | 2 | 0 → 0 | 2 → 1 |
| astropy__astropy-17800 | 0 | 0 → 0 | 0 → 4 |
| matplotlib__matplotlib-23130 | 0 | 0 → 0 | 0 → 3 |
| matplotlib__matplotlib-24674 | 1 | 1 → 1 | 2 → 4 |
| matplotlib__matplotlib-27750 | 1 | 0 → 1 | 0 → 7 |
| pydata__xarray-10006 | 0 | 0 → 0 | 1 → 0 |
| pydata__xarray-6240 | 0 | 0 → 0 | 1 → 0 |
| pydata__xarray-6260 | 2 | 1 → 1 | 1 → 2 |
| pydata__xarray-6316 | 0 | 0 → 0 | 2 → 0 |
| pydata__xarray-6710 | 0 | 0 → 0 | 5 → 2 |
| pydata__xarray-6939 | 1 | 1 → 1 | 1 → 0 |
| pydata__xarray-7489 | 0 | 0 → 0 | 1 → 2 |
| pydata__xarray-8246 | 3 | 2 → 0 | 8 → 5 |
| pydata__xarray-8441 | 0 | 0 → 0 | 5 → 10 |
| pylint-dev__pylint-6775 | 0 | 0 → 0 | 0 → 5 |
| pylint-dev__pylint-7257 | 1 | 0 → 0 | 1 → 2 |
| pylint-dev__pylint-7537 | 0 | 0 → 0 | 0 → 2 |
| pylint-dev__pylint-7918 | 1 | 0 → 0 | 0 → 7 |
| pylint-dev__pylint-8085 | 0 | 0 → 0 | 0 → 9 |
| pylint-dev__pylint-8706 | 3 | 0 → 0 | 6 → 5 |
| pylint-dev__pylint-9812 | 0 | 0 → 0 | 3 → 4 |
| pytest-dev__pytest-12232 | 3 | 1 → 1 | 0 → 5 |
| scikit-learn__scikit-learn-22957 | 2 | 0 → 0 | 2 → 1 |
| scikit-learn__scikit-learn-23459 | 2 | 0 → 0 | 2 → 1 |
| scikit-learn__scikit-learn-23937 | 0 | 0 → 0 | 5 → 6 |
| scikit-learn__scikit-learn-24456 | 0 | 0 → 0 | 1 → 0 |
| scikit-learn__scikit-learn-25108 | 4 | 1 → 0 | 2 → 2 |
| scikit-learn__scikit-learn-25148 | 9 | 1 → 1 | 2 → 3 |
| scikit-learn__scikit-learn-26052 | 0 | 0 → 0 | 1 → 0 |
| scikit-learn__scikit-learn-26420 | 0 | 0 → 0 | 3 → 1 |
| scikit-learn__scikit-learn-26928 | 2 | 0 → 1 | 1 → 3 |
| scikit-learn__scikit-learn-27943 | 1 | 0 → 0 | 0 → 2 |
| scikit-learn__scikit-learn-28564 | 2 | 0 → 1 | 0 → 0 |
| scikit-learn__scikit-learn-29916 | 0 | 0 → 0 | 0 → 1 |
| scikit-learn__scikit-learn-30296 | 0 | 0 → 0 | 0 → 2 |
| scikit-learn__scikit-learn-30581 | 1 | 1 → 1 | 2 → 1 |
| scikit-learn__scikit-learn-30727 | 0 | 0 → 0 | 1 → 3 |
| scikit-learn__scikit-learn-31106 | 0 | 0 → 0 | 4 → 15 |
| sphinx-doc__sphinx-12491 | 1 | 0 → 0 | 5 → 4 |
| sympy__sympy-22957 | 0 | 0 → 0 | 1 → 0 |
| sympy__sympy-22973 | 0 | 0 → 0 | 4 → 2 |
| sympy__sympy-23335 | 0 | 0 → 0 | 4 → 1 |
| sympy__sympy-24563 | 0 | 0 → 0 | 1 → 7 |
| sympy__sympy-24830 | 0 | 0 → 0 | 2 → 0 |
| sympy__sympy-25400 | 0 | 0 → 0 | 0 → 15 |
| sympy__sympy-25735 | 3 | 0 → 0 | 1 → 0 |
| sympy__sympy-26233 | 3 | 1 → 1 | 3 → 6 |
| sympy__sympy-27247 | 0 | 0 → 0 | 4 → 1 |
| sympy__sympy-27688 | 0 | 0 → 0 | 2 → 3 |
