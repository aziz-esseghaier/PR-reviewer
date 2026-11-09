# Review run: candidates-gpt-oss-120b-v2

- Reviewer: candidates, model openai/gpt-oss-120b (profile reviewer-gpt-oss-120b, temperature 0.0, reasoning effort medium, replies up to 3,000 tokens)
- PRs: the 26 dev PRs from 2022 on
- Reviewed: 26; failed (no usable reply in 3 tries): 0; not reviewed yet: 0
- Size limit: 10,297 characters of request text (PR, warnings, lookups, context) per request; 4 PRs trimmed

## Findings

119 findings, 4.58 per reviewed PR

- PRs with no finding (reviewed as "looks good"): 0
- Functional (F): 57; evolvability (E): 62

| Category | Name | Findings |
|---|---|---|
| E.1 | Documentation | 7 |
| E.1.1 | Documentation: text (comments, names) | 32 |
| E.1.2 | Documentation: language features | 7 |
| E.2 | Visual representation (layout, style) | 5 |
| E.3 | Structure | 1 |
| E.3.1 | Structure: organization | 4 |
| E.3.2 | Structure: solution approach | 6 |
| F.1 | Interface | 10 |
| F.2 | Logic | 16 |
| F.3 | Resource | 2 |
| F.4 | Check | 21 |
| F.5 | Support | 8 |

## Trimmed PRs

| PR | File diffs left out |
|---|---|
| astropy__astropy-16735 | 0 |
| matplotlib__matplotlib-27750 | 0 |
| pydata__xarray-8246 | 1 |
| sphinx-doc__sphinx-12491 | 3 |

## Context

- PRs whose request held context: 14 of 26; 5 had no room left after the PR's text, 7 had room but no piece (none found, or none small enough)
- Characters of context: median 3,917, largest 8,192, total 50,488

| Kind | Pieces in | Pieces left out |
|---|---|---|
| changed | 12 | 15 |
| called | 37 | 84 |
| caller | 9 | 24 |
| test | 3 | 14 |
| changed_test | 2 | 30 |
| file | 1 | 11 |

## Candidate stage

- Prompt variant: v2; candidates per PR: median 5, most 10
- Static tools: ran on 18 PRs, no Python line changed on 8, failed on 0; 8 warnings on changed lines in 2 PRs, 8 of them shown to the model
- Lookups: allowed on 22 PRs (the others had no room left), asked on 7; 11 lookups (2 read_file, 9 search_symbol), 10 found something; 0 candidates that came with lookups were dropped for the final reply
- Malformed candidates dropped from accepted replies: 0

## Model use

- Model calls: 33 (0 PRs needed more than one try)
- Tokens: 105,066 in, 47,431 out (thinking included)
