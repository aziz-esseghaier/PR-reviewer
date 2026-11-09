# Review run: candidates-gpt-oss-120b-v1

- Reviewer: candidates, model openai/gpt-oss-120b (profile reviewer-gpt-oss-120b, temperature 0.0, reasoning effort medium, replies up to 3,000 tokens)
- PRs: the 26 dev PRs from 2022 on
- Reviewed: 26; failed (no usable reply in 3 tries): 0; not reviewed yet: 0
- Size limit: 10,950 characters of request text (PR, warnings, lookups, context) per request; 4 PRs trimmed

## Findings

83 findings, 3.19 per reviewed PR

- PRs with no finding (reviewed as "looks good"): 4
- Functional (F): 56; evolvability (E): 27

| Category | Name | Findings |
|---|---|---|
| E.1 | Documentation | 3 |
| E.1.1 | Documentation: text (comments, names) | 19 |
| E.1.2 | Documentation: language features | 3 |
| E.2 | Visual representation (layout, style) | 1 |
| E.3.1 | Structure: organization | 1 |
| F.1 | Interface | 10 |
| F.2 | Logic | 16 |
| F.3 | Resource | 2 |
| F.4 | Check | 19 |
| F.5 | Support | 8 |
| F.6 | Larger defects | 1 |

## Trimmed PRs

| PR | File diffs left out |
|---|---|
| astropy__astropy-16735 | 0 |
| matplotlib__matplotlib-27750 | 0 |
| pydata__xarray-8246 | 1 |
| sphinx-doc__sphinx-12491 | 3 |

## Context

- PRs whose request held context: 17 of 26; 2 had no room left after the PR's text, 7 had room but no piece (none found, or none small enough)
- Characters of context: median 3,709, largest 8,808, total 66,559

| Kind | Pieces in | Pieces left out |
|---|---|---|
| changed | 17 | 10 |
| called | 38 | 83 |
| caller | 11 | 22 |
| test | 3 | 14 |
| changed_test | 2 | 30 |
| file | 1 | 11 |

## Candidate stage

- Prompt variant: v1; candidates per PR: median 3, most 8
- Static tools: ran on 18 PRs, no Python line changed on 8, failed on 0; 0 warnings on changed lines in 0 PRs, 0 of them shown to the model
- Lookups: allowed on 23 PRs (the others had no room left), asked on 3; 3 lookups (1 read_file, 2 search_symbol), 2 found something; 0 candidates that came with lookups were dropped for the final reply

## Model use

- Model calls: 30 (2 PRs needed more than one try)
- Tokens: 91,585 in, 39,228 out (thinking included)
