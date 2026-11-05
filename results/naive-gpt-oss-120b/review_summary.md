# Review run: naive-gpt-oss-120b

- Reviewer: naive, model openai/gpt-oss-120b (profile reviewer-gpt-oss-120b, temperature 0.0, reasoning effort medium, replies up to 3,000 tokens)
- PRs: the 68 dev PRs from 2022 on
- Reviewed: 68; failed (no usable reply in 3 tries): 0; not reviewed yet: 0
- Size limit: 11,959 characters of PR text per request; 11 PRs trimmed

## Findings

166 findings, 2.44 per reviewed PR

- PRs with no finding (reviewed as "looks good"): 20
- Functional (F): 110; evolvability (E): 56

| Category | Name | Findings |
|---|---|---|
| E.1 | Documentation | 17 |
| E.1.1 | Documentation: text (comments, names) | 29 |
| E.1.2 | Documentation: language features | 1 |
| E.2 | Visual representation (layout, style) | 4 |
| E.3 | Structure | 1 |
| E.3.1 | Structure: organization | 4 |
| F.1 | Interface | 42 |
| F.2 | Logic | 39 |
| F.3 | Resource | 6 |
| F.4 | Check | 18 |
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

## Model use

- Model calls: 69 (1 PRs needed more than one try)
- Tokens: 148,767 in, 78,844 out (thinking included)
