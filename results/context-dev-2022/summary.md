# Context retrieval: dev PRs from 2022 on

- PRs: 68; retrieved: 68
- Indexer image: pr-reviewer-indexer:b6cc5f849735; probe version 1
- Caps: 5 callers and 3 tests per changed function; called definitions, callers and tests up to 30 lines shown whole; pieces over 11,959 characters skipped
- Time per PR (probe only): median 5 s, longest 54 s

## Calls in the changed functions

- 4,629 calls: 1,623 resolved by ast (35%), 1,092 by Jedi (24%), 603 unresolved (13%), 1,311 outside the repo (28%)
- Jedi: 1,695 questions, 0 timed out, 4 failed
- Callers found 1,184, kept 162; tests found 1,212, kept 75; uses refused as ambiguous 23,192

## Per PR

| PR | Changed functions | changed | called | caller | test | changed_test | file | ast | Jedi | unresolved | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| astropy__astropy-12747 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| astropy__astropy-16735 | 4 | 2 | 6 | 0 | 0 | 1 | 0 | 7 | 5 | 11 | 10 |
| astropy__astropy-17268 | 1 | 1 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 7 |
| matplotlib__matplotlib-24674 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 13 |
| matplotlib__matplotlib-25737 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| matplotlib__matplotlib-27750 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| pydata__xarray-6260 | 2 | 1 | 4 | 1 | 0 | 0 | 0 | 3 | 3 | 1 | 5 |
| pydata__xarray-6939 | 3 | 3 | 10 | 0 | 0 | 0 | 0 | 9 | 5 | 0 | 7 |
| pydata__xarray-8246 | 10 | 0 | 24 | 0 | 0 | 8 | 1 | 73 | 29 | 11 | 9 |
| pylint-dev__pylint-7257 | 1 | 1 | 3 | 1 | 0 | 0 | 1 | 2 | 3 | 0 | 3 |
| pylint-dev__pylint-7918 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| pylint-dev__pylint-8706 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| pytest-dev__pytest-12232 | 2 | 1 | 5 | 0 | 0 | 0 | 0 | 3 | 8 | 1 | 3 |
| scikit-learn__scikit-learn-22454 | 5 | 2 | 7 | 11 | 1 | 1 | 1 | 4 | 8 | 4 | 9 |
| scikit-learn__scikit-learn-22729 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| scikit-learn__scikit-learn-22957 | 2 | 0 | 2 | 0 | 0 | 2 | 0 | 6 | 0 | 1 | 3 |
| scikit-learn__scikit-learn-23459 | 9 | 4 | 9 | 7 | 4 | 4 | 1 | 20 | 11 | 5 | 7 |
| scikit-learn__scikit-learn-25108 | 1 | 1 | 2 | 5 | 3 | 0 | 1 | 2 | 0 | 0 | 5 |
| scikit-learn__scikit-learn-25148 | 3 | 1 | 5 | 0 | 0 | 0 | 0 | 7 | 5 | 2 | 6 |
| scikit-learn__scikit-learn-26928 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-27943 | 2 | 2 | 5 | 3 | 3 | 0 | 2 | 4 | 3 | 0 | 6 |
| scikit-learn__scikit-learn-28564 | 1 | 1 | 3 | 5 | 3 | 0 | 0 | 5 | 1 | 0 | 6 |
| scikit-learn__scikit-learn-30581 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| sphinx-doc__sphinx-12491 | 22 | 3 | 11 | 0 | 0 | 15 | 3 | 106 | 5 | 41 | 6 |
| sympy__sympy-25735 | 3 | 1 | 3 | 0 | 0 | 0 | 2 | 5 | 1 | 4 | 20 |
| sympy__sympy-26233 | 2 | 1 | 22 | 0 | 0 | 1 | 0 | 99 | 10 | 6 | 24 |
| astropy__astropy-14111 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| astropy__astropy-14854 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| astropy__astropy-15549 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 7 | 3 | 6 |
| astropy__astropy-16539 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| astropy__astropy-17800 | 4 | 2 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 12 | 8 |
| django__django-15910 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| matplotlib__matplotlib-23130 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| matplotlib__matplotlib-28677 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3 |
| pydata__xarray-6240 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| pydata__xarray-6316 | 38 | 23 | 65 | 17 | 9 | 15 | 5 | 105 | 87 | 59 | 16 |
| pydata__xarray-6710 | 2 | 2 | 9 | 1 | 0 | 0 | 0 | 7 | 3 | 0 | 7 |
| pydata__xarray-7489 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 |
| pydata__xarray-8441 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 |
| pydata__xarray-10006 | 2 | 2 | 7 | 5 | 2 | 0 | 1 | 11 | 4 | 9 | 18 |
| pylint-dev__pylint-6208 | 6 | 6 | 18 | 2 | 2 | 0 | 3 | 9 | 14 | 3 | 3 |
| pylint-dev__pylint-6775 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| pylint-dev__pylint-7537 | 2 | 1 | 6 | 1 | 0 | 0 | 1 | 5 | 4 | 0 | 2 |
| pylint-dev__pylint-8085 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| pylint-dev__pylint-9812 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| pytest-dev__pytest-12746 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| scikit-learn__scikit-learn-22983 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3 |
| scikit-learn__scikit-learn-23937 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| scikit-learn__scikit-learn-24456 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 3 |
| scikit-learn__scikit-learn-26052 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| scikit-learn__scikit-learn-26420 | 21 | 11 | 24 | 25 | 20 | 10 | 0 | 39 | 40 | 33 | 8 |
| scikit-learn__scikit-learn-28330 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| scikit-learn__scikit-learn-29916 | 1 | 1 | 6 | 1 | 0 | 0 | 0 | 8 | 0 | 0 | 6 |
| scikit-learn__scikit-learn-30296 | 1 | 0 | 2 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-30465 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-30727 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-31106 | 5 | 5 | 9 | 0 | 0 | 0 | 3 | 8 | 5 | 4 | 13 |
| sympy__sympy-22957 | 1 | 1 | 12 | 1 | 0 | 0 | 1 | 27 | 2 | 9 | 35 |
| sympy__sympy-22973 | 8 | 2 | 26 | 3 | 3 | 2 | 3 | 26 | 59 | 19 | 49 |
| sympy__sympy-23335 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 23 |
| sympy__sympy-23450 | 5 | 3 | 9 | 1 | 0 | 0 | 1 | 4 | 9 | 3 | 44 |
| sympy__sympy-24161 | 4 | 4 | 7 | 7 | 5 | 0 | 0 | 10 | 3 | 2 | 36 |
| sympy__sympy-24563 | 28 | 0 | 82 | 3 | 0 | 0 | 0 | 611 | 47 | 116 | 37 |
| sympy__sympy-24830 | 32 | 19 | 54 | 0 | 4 | 10 | 4 | 165 | 285 | 150 | 29 |
| sympy__sympy-25400 | 48 | 2 | 28 | 0 | 0 | 7 | 1 | 52 | 320 | 6 | 26 |
| sympy__sympy-26660 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 12 |
| sympy__sympy-27247 | 47 | 40 | 116 | 36 | 10 | 0 | 4 | 172 | 100 | 63 | 54 |
| sympy__sympy-27688 | 2 | 1 | 5 | 0 | 0 | 1 | 0 | 7 | 5 | 23 | 28 |
