# Commit check: dev set

- 130 PRs in 11 repos, checked 2026-10-30 in sandboxes from pr-reviewer-sandbox:4703d89e263f
- Network off in every sandbox: connections to github.com and to the proxy failed before each repo's checks, as they should
- Base: the commit just before each PR's first reviewed commit (found with git). PR state: its last reviewed commit

## Usable PRs

A PR is usable when all its reviewed commits are in the shared copy and both sides were checked out and ran a command.

| | PRs |
|---|---|
| Usable | 130 |
| Not usable | 0 |

## The dataset's base_commit field, seen from the base

| base_commit field is ... | PRs |
|---|---|
| the base itself | 75 |
| later on the same line (other commits came in between) | 55 |
| earlier on the same line | 0 |
| on another line of history | 0 |
| not in the repo | 0 |

When it's later, it's 7 commits after the base in the middle case, and up to 3317.

## Merges and extra changes

- PRs with a merge among their reviewed commits: 1
- PRs where base vs PR state also changes files no reviewed commit changed: 1

## By repo

| Repo | PRs | Usable | base_commit field = base | With a merge | With extra files |
|---|---|---|---|---|---|
| astropy/astropy | 11 | 11 | 10 | 0 | 0 |
| django/django | 1 | 1 | 1 | 1 | 0 |
| matplotlib/matplotlib | 12 | 12 | 9 | 0 | 1 |
| mwaskom/seaborn | 2 | 2 | 2 | 0 | 0 |
| psf/requests | 1 | 1 | 0 | 0 | 0 |
| pydata/xarray | 14 | 14 | 8 | 0 | 0 |
| pylint-dev/pylint | 11 | 11 | 6 | 0 | 0 |
| pytest-dev/pytest | 6 | 6 | 4 | 0 | 0 |
| scikit-learn/scikit-learn | 38 | 38 | 20 | 0 | 0 |
| sphinx-doc/sphinx | 3 | 3 | 0 | 0 | 0 |
| sympy/sympy | 31 | 31 | 15 | 0 | 0 |

## PRs to look at

- matplotlib__matplotlib-17851: 2 extra file(s), such as lib/mpl_toolkits/tests/baseline_images/test_mplot3d/add_collection3d_zs_array.png
- django__django-15910: 1 merge commit(s)
