# Environment setup: dev set

- 68 of 68 dev PRs from 2022 onward set up so far; agent model setup-gpt-oss-120b, at most 12 model calls and 60 minutes each
- Verified: the related tests (at most 5 files) passed, at least 90% of them, at the PR's base and again at its PR state, with the network off

## Overall

Verified: 68/68 (100%)

## By repo

| | Verified |
|---|---|
| astropy/astropy | 8/8 (100%) |
| django/django | 1/1 (100%) |
| matplotlib/matplotlib | 5/5 (100%) |
| pydata/xarray | 9/9 (100%) |
| pylint-dev/pylint | 8/8 (100%) |
| pytest-dev/pytest | 2/2 (100%) |
| scikit-learn/scikit-learn | 21/21 (100%) |
| sphinx-doc/sphinx | 1/1 (100%) |
| sympy/sympy | 13/13 (100%) |

## By year

| | Verified |
|---|---|
| 2022 | 29/29 (100%) |
| 2023 | 16/16 (100%) |
| 2024 | 17/17 (100%) |
| 2025 | 6/6 (100%) |

## What got environments working

| Source | PRs |
|---|---|
| a recipe kept from an earlier PR | 51 |
| the standard recipe | 14 |
| a recipe from the setup agent's model | 3 |

26 of the verified PRs change no Python code, so they were checked with a few of the repo's own tests instead of related ones.

## Why setups failed

| Failure category | PRs |
|---|---|
| (none) | 0 |

Model calls in all: 14. Minutes in sandboxes: 429.

The usable subset (usable_prs.json) holds the 68 verified PRs.
