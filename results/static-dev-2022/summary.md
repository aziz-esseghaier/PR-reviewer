# Static tools: dev PRs from 2022 on

- PRs: 68; tools ran on 47; no Python line changed 21; failed 0; not run yet 0
- Tools image: pr-reviewer-tools:f9bab73f9ba7
- Versions: ruff 0.17.4; mypy 2.4.1 (compiled: yes); bandit 1.9.4; semgrep 1.184.0; semgrep-rules 16b9e6d45f10649f6a531ec4838ed9924372c3bc 284 rule files
- Rules: ruff F,B,PLE; mypy defaults (--ignore-missing-imports --check-untyped-defs); bandit severity medium and up; semgrep correctness and security rules for Python. Repo configs ignored.
- Each tool runs at most 10 minutes per PR, network off, in the PR's prepared image at its last commit

## Per tool

| Tool | Finished | Timed out | Out of memory | Failed | Warnings on changed files | On changed lines | PRs with one |
|---|---|---|---|---|---|---|---|
| ruff | 47 | 0 | 0 | 0 | 468 | 4 | 3 |
| mypy | 47 | 0 | 0 | 0 | 2255 | 27 | 6 |
| bandit | 47 | 0 | 0 | 0 | 25 | 0 | 0 |
| semgrep | 47 | 0 | 0 | 0 | 83 | 1 | 1 |

mypy followed imports silently on 47 PRs and checked the changed files alone on 0 (following imports ran out of time or memory).

## Rules on changed lines

| Tool | Rule | Warnings |
|---|---|---|
| mypy | index | 6 |
| mypy | operator | 5 |
| mypy | assignment | 5 |
| mypy | override | 3 |
| mypy | arg-type | 2 |
| ruff | B007 | 2 |
| mypy | attr-defined | 2 |
| mypy | call-arg | 2 |
| mypy | misc | 1 |
| mypy | valid-type | 1 |
| ruff | F632 | 1 |
| semgrep | list-modify-while-iterate | 1 |
| ruff | F401 | 1 |

## Per PR

| PR | Files | ruff | mypy | bandit | semgrep | On the files | mypy imports | Seconds |
|---|---|---|---|---|---|---|---|---|
| astropy__astropy-12747 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 54 |
| astropy__astropy-16735 | 2 | 0 | 0 | 0 | 0 | 55 | silent | 110 |
| astropy__astropy-17268 | 1 | 0 | 0 | 0 | 0 | 53 | silent | 60 |
| matplotlib__matplotlib-24674 | 1 | 0 | 0 | 0 | 0 | 54 | silent | 105 |
| matplotlib__matplotlib-25737 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 29 |
| matplotlib__matplotlib-27750 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 12 |
| pydata__xarray-6260 | 2 | 0 | 0 | 0 | 0 | 94 | silent | 70 |
| pydata__xarray-6939 | 3 | 0 | 0 | 0 | 0 | 110 | silent | 62 |
| pydata__xarray-8246 | 4 | 0 | 0 | 0 | 0 | 144 | silent | 100 |
| pylint-dev__pylint-7257 | 2 | 0 | 0 | 0 | 0 | 1 | silent | 54 |
| pylint-dev__pylint-7918 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 19 |
| pylint-dev__pylint-8706 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 28 |
| pytest-dev__pytest-12232 | 2 | 0 | 0 | 0 | 0 | 12 | silent | 52 |
| scikit-learn__scikit-learn-22454 | 5 | 0 | 0 | 0 | 0 | 35 | silent | 43 |
| scikit-learn__scikit-learn-22729 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 10 |
| scikit-learn__scikit-learn-22957 | 1 | 0 | 0 | 0 | 0 | 1 | silent | 46 |
| scikit-learn__scikit-learn-23459 | 3 | 0 | 0 | 0 | 0 | 23 | silent | 64 |
| scikit-learn__scikit-learn-25108 | 2 | 0 | 0 | 0 | 0 | 8 | silent | 69 |
| scikit-learn__scikit-learn-25148 | 2 | 0 | 0 | 0 | 0 | 5 | silent | 57 |
| scikit-learn__scikit-learn-26928 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 14 |
| scikit-learn__scikit-learn-27943 | 2 | 0 | 0 | 0 | 0 | 1 | silent | 66 |
| scikit-learn__scikit-learn-28564 | 1 | 0 | 0 | 0 | 0 | 24 | silent | 91 |
| scikit-learn__scikit-learn-30581 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 18 |
| sphinx-doc__sphinx-12491 | 7 | 0 | 6 | 0 | 0 | 45 | silent | 81 |
| sympy__sympy-25735 | 2 | 0 | 0 | 0 | 0 | 4 | silent | 86 |
| sympy__sympy-26233 | 2 | 0 | 2 | 0 | 0 | 84 | silent | 72 |
| astropy__astropy-14111 | 1 | 0 | 0 | 0 | 0 | 36 | silent | 60 |
| astropy__astropy-14854 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 20 |
| astropy__astropy-15549 | 1 | 0 | 2 | 0 | 0 | 9 | silent | 30 |
| astropy__astropy-16539 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 7 |
| astropy__astropy-17800 | 2 | 0 | 5 | 0 | 0 | 106 | silent | 30 |
| django__django-15910 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 23 |
| matplotlib__matplotlib-23130 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 11 |
| matplotlib__matplotlib-28677 | 1 | 0 | 0 | 0 | 0 | 0 | silent | 34 |
| pydata__xarray-6240 | 1 | 0 | 0 | 0 | 0 | 5 | silent | 36 |
| pydata__xarray-6316 | 26 | 0 | 0 | 0 | 0 | 783 | silent | 69 |
| pydata__xarray-6710 | 2 | 0 | 0 | 0 | 0 | 14 | silent | 28 |
| pydata__xarray-7489 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 4 |
| pydata__xarray-8441 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 3 |
| pydata__xarray-10006 | 2 | 0 | 0 | 0 | 0 | 10 | silent | 45 |
| pylint-dev__pylint-6208 | 3 | 0 | 0 | 0 | 0 | 2 | silent | 30 |
| pylint-dev__pylint-6775 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 6 |
| pylint-dev__pylint-7537 | 2 | 1 | 0 | 0 | 0 | 43 | silent | 33 |
| pylint-dev__pylint-8085 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 8 |
| pylint-dev__pylint-9812 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 10 |
| pytest-dev__pytest-12746 | 3 | 0 | 0 | 0 | 0 | 12 | silent | 38 |
| scikit-learn__scikit-learn-22983 | 1 | 0 | 0 | 0 | 0 | 0 | silent | 57 |
| scikit-learn__scikit-learn-23937 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 18 |
| scikit-learn__scikit-learn-24456 | 4 | 0 | 0 | 0 | 0 | 0 | silent | 26 |
| scikit-learn__scikit-learn-26052 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 4 |
| scikit-learn__scikit-learn-26420 | 6 | 0 | 0 | 0 | 0 | 25 | silent | 30 |
| scikit-learn__scikit-learn-28330 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 4 |
| scikit-learn__scikit-learn-29916 | 1 | 0 | 0 | 0 | 0 | 7 | silent | 29 |
| scikit-learn__scikit-learn-30296 | 1 | 0 | 0 | 0 | 0 | 20 | silent | 31 |
| scikit-learn__scikit-learn-30465 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 3 |
| scikit-learn__scikit-learn-30727 | 1 | 0 | 0 | 0 | 0 | 27 | silent | 30 |
| scikit-learn__scikit-learn-31106 | 5 | 0 | 0 | 0 | 0 | 50 | silent | 33 |
| sympy__sympy-22957 | 1 | 0 | 0 | 0 | 0 | 7 | silent | 30 |
| sympy__sympy-22973 | 6 | 0 | 0 | 0 | 0 | 100 | silent | 29 |
| sympy__sympy-23335 | 1 | 0 | 0 | 0 | 0 | 5 | silent | 27 |
| sympy__sympy-23450 | 3 | 0 | 0 | 0 | 0 | 20 | silent | 31 |
| sympy__sympy-24161 | 1 | 0 | 0 | 0 | 0 | 47 | silent | 29 |
| sympy__sympy-24563 | 3 | 2 | 8 | 0 | 0 | 38 | silent | 27 |
| sympy__sympy-24830 | 6 | 0 | 4 | 0 | 0 | 10 | silent | 29 |
| sympy__sympy-25400 | 4 | 0 | 0 | 0 | 0 | 0 | silent | 27 |
| sympy__sympy-26660 | 1 | 0 | 0 | 0 | 0 | 0 | silent | 25 |
| sympy__sympy-27247 | 24 | 1 | 0 | 0 | 1 | 693 | silent | 38 |
| sympy__sympy-27688 | 2 | 0 | 0 | 0 | 0 | 9 | silent | 28 |
