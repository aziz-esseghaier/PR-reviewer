# Review run: context-gpt-oss-120b

- Reviewer: context, model openai/gpt-oss-120b (profile reviewer-gpt-oss-120b, temperature 0.0, reasoning effort medium, replies up to 3,000 tokens)
- PRs: the 68 dev PRs from 2022 on
- Reviewed: 68; failed (no usable reply in 3 tries): 0; not reviewed yet: 0
- Size limit: 11,959 characters of PR text per request; 11 PRs trimmed

## Findings

132 findings, 1.94 per reviewed PR

- PRs with no finding (reviewed as "looks good"): 23
- Functional (F): 84; evolvability (E): 48

| Category | Name | Findings |
|---|---|---|
| E.1 | Documentation | 15 |
| E.1.1 | Documentation: text (comments, names) | 25 |
| E.1.2 | Documentation: language features | 1 |
| E.2 | Visual representation (layout, style) | 4 |
| E.3 | Structure | 1 |
| E.3.1 | Structure: organization | 2 |
| F.1 | Interface | 25 |
| F.2 | Logic | 29 |
| F.3 | Resource | 4 |
| F.4 | Check | 21 |
| F.5 | Support | 5 |

## Trimmed PRs

| PR | File diffs left out |
|---|---|
| matplotlib__matplotlib-27750 | 0 |
| sphinx-doc__sphinx-12491 | 4 |
| astropy__astropy-14854 | 56 |
| pydata__xarray-6316 | 18 |
| pydata__xarray-8441 | 1 |
| scikit-learn__scikit-learn-30465 | 103 |
| scikit-learn__scikit-learn-31106 | 5 |
| sympy__sympy-24563 | 2 |
| sympy__sympy-24830 | 4 |
| sympy__sympy-25400 | 47 |
| sympy__sympy-27247 | 15 |

## Context

- PRs whose request held context: 37 of 68; 8 had no room left after the PR's text, 23 had room but no piece (none found, or none small enough)
- Characters of context: median 4,633, largest 9,917, total 178,265

| Kind | Pieces in | Pieces left out |
|---|---|---|
| changed | 39 | 113 |
| called | 124 | 485 |
| caller | 17 | 119 |
| test | 5 | 67 |
| changed_test | 4 | 75 |
| file | 7 | 42 |

## Model use

- Model calls: 71 (3 PRs needed more than one try)
- Tokens: 204,194 in, 87,290 out (thinking included)
