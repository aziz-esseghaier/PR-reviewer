# Evidence run evidence-tools-gpt-oss-120b

Candidates from `candidates-gpt-oss-120b`, checked with evidence-gpt-oss-120b (openai/gpt-oss-120b) on 2026-11-17.

## Progress

- Candidates checked: 204 of 204 (100.0%); failed (to try again): 0
- PRs: 50

## Tiers

| Tier | Functional | Evolvability | All |
|---|---|---|---|
| 1 (executed) | 5 | 0 | 5 |
| 1 (new code) | 0 | 0 | 0 |
| 2 | 1 | 3 | 4 |
| 3 | 16 | 4 | 20 |
| 4 | 126 | 49 | 175 |

## Proof rate

- Functional candidates proven by a test (tier 1): 5 of 148 (3.4%)
- Testable ones (functional, naming an exception): 5 of 115 (4.3%)
- New-code proofs among them: 0

## Test writer

| How it ended | Candidates |
|---|---|
| proof | 5 |
| no_proof | 40 |
| broken | 29 |
| budget | 0 |
| invalid | 6 |
| no_reply | 3 |
| not_tried | 121 |

- Tests written: 157; refused by our checks before running: 51; broken when run: 61
- Why runs proved nothing: passes on both commits (20); the test's own code raised the error (7); new code: passes on the PR (4); the PR run failed with Failed (2); the PR run failed with AssertionError (2); new code: the error didn't come through a line the PR changed (1); the error didn't come through a line the PR changed (1); the PR run failed with TypeError (1)

## Verifier

| Verdict | Hit (matches a known issue) | False alarm on a PR with issues | False alarm on a clean PR | Not scored |
|---|---|---|---|---|
| upheld | 0 | 7 | 13 | 0 |
| refuted | 10 | 59 | 110 | 0 |
| not asked | 2 | 3 | 0 | 0 |

- Static warnings confirmed as the same problem: 4 candidates

## Tier against the judge

Each candidate's tier, against how the judge saw the same finding in the source run.

| Tier | Hit (matches a known issue) | False alarm on a PR with issues | False alarm on a clean PR | Not scored |
|---|---|---|---|---|
| 1 (executed) | 2 | 3 | 0 | 0 |
| 1 (new code) | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 4 | 0 |
| 3 | 0 | 7 | 13 | 0 |
| 4 | 10 | 59 | 106 | 0 |

## Cost

- Model replies: 574 (2.8 per candidate)
- Tokens: 864,599 in, 294,750 out (5,683 per candidate)
- Test runs in Docker: 84 minutes (stored runs count 0)

## Proven findings

| PR | Finding | Kind | Claim | Judge | Test |
|---|---|---|---|---|---|
| pydata__xarray-6939 | f1 | executed | Error message string does not interpolate the method name | hit | proofs/pydata__xarray-6939__f1.py |
| scikit-learn__scikit-learn-25148 | f1 | executed | predict_proba does not validate input X | false_alarm | proofs/scikit-learn__scikit-learn-25148__f1.py |
| scikit-learn__scikit-learn-27943 | f2 | executed | Changing default solver may alter model behavior for existing code | false_alarm | proofs/scikit-learn__scikit-learn-27943__f2.py |
| sphinx-doc__sphinx-12491 | f1 | executed | _lift_sections may raise AttributeError when state_parent.parent is None | false_alarm | proofs/sphinx-doc__sphinx-12491__f1.py |
| sympy__sympy-26233 | f7 | executed | I*pi term placement may affect the series sign | hit | proofs/sympy__sympy-26233__f7.py |
