# Context retrieval: test PRs from 2022 on

- PRs: 153; retrieved: 153
- Indexer image: pr-reviewer-indexer:b6cc5f849735; probe version 1
- Caps: 5 callers and 3 tests per changed function; called definitions, callers and tests up to 30 lines shown whole; pieces over 11,959 characters skipped
- Time per PR (probe only): median 7 s, longest 75 s

## Calls in the changed functions

- 10,257 calls: 1,998 resolved by ast (19%), 2,623 by Jedi (26%), 1,789 unresolved (17%), 3,847 outside the repo (38%)
- Jedi: 4,412 questions, 7 timed out, 15 failed
- Callers found 490, kept 234; tests found 1,163, kept 96; uses refused as ambiguous 73,951

## Per PR

| PR | Changed functions | changed | called | caller | test | changed_test | file | ast | Jedi | unresolved | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| astropy__astropy-14682 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |
| astropy__astropy-15492 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| astropy__astropy-16207 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| astropy__astropy-16872 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| astropy__astropy-17461 | 5 | 4 | 6 | 3 | 3 | 0 | 0 | 5 | 4 | 1 | 22 |
| astropy__astropy-17961 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| django__django-16040 | 1 | 1 | 4 | 1 | 0 | 0 | 0 | 6 | 2 | 30 | 22 |
| django__django-18803 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |
| matplotlib__matplotlib-22696 | 1 | 1 | 6 | 2 | 2 | 0 | 0 | 2 | 4 | 0 | 7 |
| matplotlib__matplotlib-23027 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 2 | 4 | 7 |
| matplotlib__matplotlib-23066 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| matplotlib__matplotlib-25692 | 2 | 1 | 5 | 0 | 0 | 0 | 0 | 2 | 3 | 1 | 19 |
| matplotlib__matplotlib-25993 | 1 | 1 | 3 | 1 | 0 | 0 | 1 | 1 | 4 | 1 | 8 |
| matplotlib__matplotlib-27128 | 3 | 0 | 4 | 0 | 0 | 2 | 1 | 8 | 2 | 4 | 6 |
| matplotlib__matplotlib-27267 | 5 | 1 | 4 | 1 | 0 | 1 | 1 | 5 | 0 | 2 | 15 |
| matplotlib__matplotlib-28625 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| matplotlib__matplotlib-29725 | 7 | 0 | 9 | 0 | 0 | 0 | 1 | 6 | 5 | 2 | 6 |
| pydata__xarray-6237 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| pydata__xarray-6461 | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 3 |
| pydata__xarray-6592 | 2 | 1 | 7 | 2 | 0 | 1 | 0 | 5 | 3 | 2 | 7 |
| pydata__xarray-7798 | 3 | 1 | 10 | 0 | 0 | 1 | 0 | 7 | 26 | 6 | 11 |
| pydata__xarray-8287 | 9 | 6 | 7 | 0 | 0 | 3 | 0 | 11 | 6 | 0 | 9 |
| pydata__xarray-8535 | 2 | 0 | 17 | 0 | 0 | 0 | 0 | 16 | 6 | 15 | 9 |
| pylint-dev__pylint-6367 | 2 | 1 | 6 | 5 | 0 | 1 | 1 | 6 | 0 | 0 | 2 |
| pylint-dev__pylint-6758 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| pylint-dev__pylint-6782 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 4 |
| pylint-dev__pylint-6879 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| pylint-dev__pylint-7862 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| pylint-dev__pylint-8926 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| pylint-dev__pylint-9016 | 2 | 1 | 1 | 1 | 2 | 1 | 2 | 1 | 0 | 1 | 4 |
| pylint-dev__pylint-9316 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| pylint-dev__pylint-9339 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| pytest-dev__pytest-11659 | 29 | 2 | 27 | 1 | 0 | 27 | 0 | 66 | 24 | 9 | 6 |
| pytest-dev__pytest-13293 | 4 | 2 | 3 | 2 | 2 | 1 | 0 | 6 | 6 | 13 | 4 |
| scikit-learn__scikit-learn-22306 | 1 | 0 | 8 | 0 | 0 | 1 | 0 | 1 | 13 | 0 | 5 |
| scikit-learn__scikit-learn-22412 | 2 | 1 | 3 | 5 | 3 | 0 | 0 | 1 | 3 | 2 | 7 |
| scikit-learn__scikit-learn-22862 | 4 | 0 | 1 | 0 | 0 | 4 | 0 | 2 | 0 | 1 | 4 |
| scikit-learn__scikit-learn-22950 | 8 | 5 | 20 | 1 | 3 | 2 | 0 | 25 | 14 | 8 | 10 |
| scikit-learn__scikit-learn-23066 | 1 | 0 | 3 | 0 | 0 | 1 | 0 | 1 | 8 | 0 | 4 |
| scikit-learn__scikit-learn-23106 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-23110 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-23813 | 2 | 2 | 6 | 0 | 0 | 0 | 0 | 4 | 6 | 0 | 7 |
| scikit-learn__scikit-learn-24883 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| scikit-learn__scikit-learn-24898 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 5 |
| scikit-learn__scikit-learn-24928 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-24948 | 1 | 1 | 2 | 2 | 0 | 0 | 1 | 0 | 2 | 0 | 7 |
| scikit-learn__scikit-learn-25474 | 1 | 1 | 5 | 5 | 3 | 0 | 1 | 14 | 1 | 4 | 9 |
| scikit-learn__scikit-learn-26229 | 2 | 1 | 15 | 3 | 3 | 1 | 0 | 12 | 5 | 2 | 10 |
| scikit-learn__scikit-learn-26690 | 1 | 1 | 7 | 5 | 3 | 0 | 0 | 12 | 1 | 1 | 7 |
| scikit-learn__scikit-learn-27865 | 2 | 1 | 6 | 1 | 1 | 1 | 1 | 12 | 2 | 4 | 9 |
| scikit-learn__scikit-learn-28332 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| scikit-learn__scikit-learn-28599 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| scikit-learn__scikit-learn-29460 | 1 | 0 | 2 | 0 | 0 | 1 | 4 | 0 | 2 | 1 | 8 |
| scikit-learn__scikit-learn-29838 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |
| scikit-learn__scikit-learn-30373 | 2 | 1 | 9 | 0 | 1 | 1 | 0 | 10 | 18 | 0 | 7 |
| scikit-learn__scikit-learn-30726 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 5 |
| scikit-learn__scikit-learn-31119 | 7 | 6 | 22 | 0 | 0 | 1 | 0 | 11 | 38 | 9 | 16 |
| sphinx-doc__sphinx-12131 | 2 | 0 | 3 | 0 | 0 | 0 | 1 | 2 | 1 | 3 | 4 |
| sympy__sympy-24012 | 12 | 3 | 44 | 0 | 0 | 2 | 0 | 225 | 70 | 13 | 57 |
| sympy__sympy-24358 | 6 | 0 | 9 | 0 | 0 | 0 | 0 | 25 | 11 | 9 | 37 |
| sympy__sympy-25105 | 27 | 6 | 13 | 0 | 4 | 2 | 2 | 12 | 42 | 2 | 31 |
| sympy__sympy-25504 | 4 | 2 | 7 | 9 | 4 | 0 | 2 | 21 | 1 | 4 | 32 |
| sympy__sympy-27396 | 3 | 1 | 31 | 0 | 0 | 0 | 0 | 56 | 23 | 75 | 46 |
| sympy__sympy-27708 | 2 | 1 | 20 | 0 | 0 | 0 | 1 | 24 | 607 | 645 | 34 |
| astropy__astropy-13841 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |
| astropy__astropy-14149 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |
| astropy__astropy-14437 | 4 | 4 | 6 | 2 | 0 | 0 | 4 | 9 | 3 | 5 | 19 |
| astropy__astropy-14607 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| astropy__astropy-14782 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| astropy__astropy-15612 | 1 | 1 | 5 | 2 | 0 | 0 | 0 | 2 | 6 | 3 | 20 |
| astropy__astropy-15723 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| astropy__astropy-16703 | 11 | 0 | 7 | 0 | 0 | 9 | 0 | 4 | 16 | 12 | 9 |
| astropy__astropy-17155 | 3 | 2 | 7 | 5 | 0 | 0 | 1 | 9 | 2 | 12 | 14 |
| astropy__astropy-17278 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 6 |
| matplotlib__matplotlib-23942 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| matplotlib__matplotlib-25853 | 5 | 5 | 11 | 3 | 0 | 0 | 0 | 13 | 6 | 16 | 75 |
| matplotlib__matplotlib-26005 | 2 | 0 | 1 | 0 | 0 | 2 | 1 | 1 | 0 | 4 | 5 |
| mwaskom__seaborn-3447 | 17 | 10 | 29 | 13 | 6 | 7 | 3 | 41 | 17 | 13 | 9 |
| pydata__xarray-6414 | 2 | 1 | 11 | 5 | 0 | 1 | 0 | 10 | 7 | 2 | 11 |
| pydata__xarray-6823 | 2 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 10 | 2 | 10 |
| pydata__xarray-6988 | 2 | 0 | 2 | 0 | 0 | 2 | 0 | 1 | 2 | 0 | 2 |
| pydata__xarray-7022 | 4 | 3 | 7 | 0 | 0 | 1 | 0 | 7 | 3 | 5 | 8 |
| pydata__xarray-7114 | 255 | 9 | 79 | 2 | 9 | 246 | 4 | 547 | 691 | 153 | 19 |
| pydata__xarray-7222 | 1 | 1 | 7 | 0 | 0 | 0 | 0 | 7 | 0 | 1 | 4 |
| pydata__xarray-7876 | 5 | 2 | 6 | 0 | 0 | 3 | 0 | 8 | 16 | 13 | 7 |
| pydata__xarray-8296 | 2 | 2 | 0 | 10 | 6 | 0 | 0 | 0 | 0 | 0 | 4 |
| pydata__xarray-8365 | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 4 |
| pydata__xarray-8438 | 20 | 5 | 31 | 0 | 3 | 13 | 0 | 32 | 74 | 32 | 17 |
| pydata__xarray-8605 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| pydata__xarray-8890 | 63 | 0 | 39 | 0 | 0 | 63 | 0 | 93 | 351 | 123 | 17 |
| pydata__xarray-10219 | 2 | 1 | 10 | 0 | 0 | 0 | 0 | 4 | 10 | 7 | 10 |
| pylint-dev__pylint-6575 | 1 | 0 | 5 | 0 | 0 | 1 | 1 | 4 | 1 | 2 | 4 |
| pylint-dev__pylint-6958 | 4 | 4 | 3 | 1 | 0 | 0 | 1 | 6 | 0 | 1 | 7 |
| pylint-dev__pylint-6969 | 1 | 0 | 2 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 2 |
| pylint-dev__pylint-7410 | 2 | 1 | 8 | 0 | 0 | 0 | 1 | 3 | 6 | 0 | 3 |
| pylint-dev__pylint-7921 | 2 | 1 | 1 | 3 | 0 | 0 | 2 | 1 | 0 | 0 | 2 |
| pylint-dev__pylint-8279 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| pylint-dev__pylint-8819 | 5 | 1 | 17 | 1 | 2 | 0 | 3 | 13 | 16 | 10 | 10 |
| pylint-dev__pylint-9195 | 3 | 3 | 21 | 1 | 0 | 0 | 0 | 23 | 6 | 2 | 4 |
| pylint-dev__pylint-9589 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| pylint-dev__pylint-9630 | 2 | 1 | 3 | 3 | 0 | 0 | 3 | 4 | 0 | 0 | 2 |
| pylint-dev__pylint-9940 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| pylint-dev__pylint-9978 | 31 | 31 | 30 | 19 | 0 | 0 | 0 | 51 | 15 | 23 | 6 |
| pytest-dev__pytest-10383 | 5 | 1 | 5 | 0 | 0 | 2 | 0 | 4 | 4 | 5 | 3 |
| pytest-dev__pytest-10678 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| scikit-learn__scikit-learn-23420 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| scikit-learn__scikit-learn-23915 | 1 | 1 | 1 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 6 |
| scikit-learn__scikit-learn-24734 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-25025 | 1 | 1 | 3 | 5 | 3 | 0 | 1 | 3 | 0 | 0 | 6 |
| scikit-learn__scikit-learn-26228 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-26633 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-26670 | 1 | 1 | 1 | 2 | 0 | 0 | 1 | 7 | 0 | 1 | 7 |
| scikit-learn__scikit-learn-27444 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-27539 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-27958 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| scikit-learn__scikit-learn-28121 | 2 | 1 | 3 | 0 | 0 | 0 | 1 | 1 | 2 | 1 | 6 |
| scikit-learn__scikit-learn-28152 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| scikit-learn__scikit-learn-28198 | 14 | 2 | 28 | 0 | 0 | 12 | 2 | 16 | 74 | 29 | 10 |
| scikit-learn__scikit-learn-28802 | 54 | 36 | 120 | 38 | 13 | 17 | 84 | 136 | 136 | 78 | 23 |
| scikit-learn__scikit-learn-28843 | 2 | 1 | 22 | 0 | 0 | 0 | 0 | 15 | 12 | 5 | 12 |
| scikit-learn__scikit-learn-29228 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| scikit-learn__scikit-learn-29357 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-29493 | 1 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 2 | 7 |
| scikit-learn__scikit-learn-30026 | 1 | 0 | 3 | 0 | 0 | 1 | 0 | 2 | 2 | 0 | 4 |
| scikit-learn__scikit-learn-30361 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-30409 | 6 | 0 | 23 | 1 | 3 | 3 | 0 | 27 | 22 | 9 | 8 |
| scikit-learn__scikit-learn-30862 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| scikit-learn__scikit-learn-31208 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| sphinx-doc__sphinx-11708 | 1 | 1 | 1 | 5 | 3 | 0 | 1 | 1 | 0 | 1 | 3 |
| sphinx-doc__sphinx-12466 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| sympy__sympy-22870 | 4 | 3 | 32 | 1 | 0 | 0 | 0 | 68 | 21 | 57 | 49 |
| sympy__sympy-22897 | 4 | 1 | 13 | 0 | 0 | 1 | 0 | 28 | 13 | 28 | 45 |
| sympy__sympy-23026 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 26 |
| sympy__sympy-23162 | 1 | 1 | 12 | 0 | 0 | 0 | 0 | 18 | 1 | 17 | 30 |
| sympy__sympy-23357 | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 30 | 1 | 1 | 32 |
| sympy__sympy-23372 | 2 | 1 | 8 | 0 | 0 | 0 | 1 | 5 | 6 | 5 | 51 |
| sympy__sympy-23659 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 28 |
| sympy__sympy-23742 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 28 |
| sympy__sympy-23824 | 2 | 0 | 4 | 1 | 0 | 1 | 0 | 38 | 1 | 123 | 49 |
| sympy__sympy-23941 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 43 |
| sympy__sympy-23976 | 2 | 1 | 9 | 0 | 0 | 1 | 0 | 11 | 1 | 12 | 49 |
| sympy__sympy-24177 | 8 | 0 | 12 | 0 | 0 | 0 | 1 | 3 | 32 | 12 | 58 |
| sympy__sympy-24501 | 22 | 10 | 0 | 15 | 5 | 7 | 2 | 9 | 2 | 19 | 28 |
| sympy__sympy-24573 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 4 | 5 | 33 |
| sympy__sympy-25074 | 3 | 2 | 11 | 0 | 0 | 1 | 2 | 25 | 39 | 21 | 29 |
| sympy__sympy-25246 | 2 | 1 | 7 | 0 | 0 | 0 | 0 | 2 | 11 | 2 | 37 |
| sympy__sympy-25549 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 15 |
| sympy__sympy-25622 | 6 | 6 | 5 | 0 | 0 | 0 | 0 | 5 | 7 | 11 | 23 |
| sympy__sympy-26435 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 25 |
| sympy__sympy-26984 | 3 | 1 | 9 | 2 | 0 | 1 | 0 | 13 | 7 | 24 | 39 |
| sympy__sympy-27242 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 |
| sympy__sympy-27243 | 1 | 0 | 2 | 0 | 0 | 1 | 0 | 2 | 0 | 1 | 25 |
| sympy__sympy-27763 | 1 | 1 | 3 | 1 | 3 | 0 | 0 | 3 | 0 | 3 | 26 |
