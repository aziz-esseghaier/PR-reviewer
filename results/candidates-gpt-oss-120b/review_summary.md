# Review run: candidates-gpt-oss-120b

- Reviewer: candidates, model openai/gpt-oss-120b (profile reviewer-gpt-oss-120b, temperature 0.0, reasoning effort medium, replies up to 3,000 tokens)
- PRs: the 68 dev PRs from 2022 on
- Reviewed: 68; failed (no usable reply in 3 tries): 0; not reviewed yet: 0
- Size limit: 10,950 characters of request text (PR, warnings, lookups, context) per request; 14 PRs trimmed

## Findings

204 findings, 3.00 per reviewed PR

- PRs with no finding (reviewed as "looks good"): 18
- Functional (F): 148; evolvability (E): 56

| Category | Name | Findings |
|---|---|---|
| E.1 | Documentation | 10 |
| E.1.1 | Documentation: text (comments, names) | 29 |
| E.1.2 | Documentation: language features | 8 |
| E.2 | Visual representation (layout, style) | 4 |
| E.3.1 | Structure: organization | 5 |
| F.1 | Interface | 37 |
| F.2 | Logic | 34 |
| F.3 | Resource | 8 |
| F.4 | Check | 43 |
| F.5 | Support | 25 |
| F.6 | Larger defects | 1 |

## Trimmed PRs

| PR | File diffs left out |
|---|---|
| astropy__astropy-16735 | 0 |
| matplotlib__matplotlib-27750 | 0 |
| pydata__xarray-8246 | 1 |
| sphinx-doc__sphinx-12491 | 3 |
| astropy__astropy-14854 | 61 |
| pydata__xarray-6316 | 19 |
| pydata__xarray-8441 | 2 |
| scikit-learn__scikit-learn-26420 | 2 |
| scikit-learn__scikit-learn-30465 | 109 |
| scikit-learn__scikit-learn-31106 | 6 |
| sympy__sympy-24563 | 2 |
| sympy__sympy-24830 | 5 |
| sympy__sympy-25400 | 47 |
| sympy__sympy-27247 | 17 |

## Context

- PRs whose request held context: 36 of 68; 9 had no room left after the PR's text, 23 had room but no piece (none found, or none small enough)
- Characters of context: median 3,574, largest 8,808, total 144,202

| Kind | Pieces in | Pieces left out |
|---|---|---|
| changed | 33 | 119 |
| called | 96 | 513 |
| caller | 13 | 123 |
| test | 3 | 69 |
| changed_test | 3 | 76 |
| file | 6 | 43 |

## Candidate stage

- Prompt variant: v1; candidates per PR: median 2, most 15
- Static tools: ran on 47 PRs, no Python line changed on 21, failed on 0; 32 warnings on changed lines in 8 PRs, 30 of them shown to the model
- Lookups: allowed on 55 PRs (the others had no room left), asked on 11; 20 lookups (8 read_file, 12 search_symbol), 12 found something; 1 candidates that came with lookups were dropped for the final reply
- Malformed candidates dropped from accepted replies: 0

## Model use

- Model calls: 86 (7 PRs needed more than one try)
- Tokens: 251,858 in, 103,079 out (thinking included)
