"""The evidence agent: check each candidate finding with a test, the static tools or a verifier.

A review run (such as the candidate reviewer's) lists candidate problems. For each one, the
agent looks for the strongest evidence it can get, as one LangGraph subgraph per candidate:
- a generated test that passes on the base commit and fails on the PR with the claimed error
  (tier 1), written and fixed by the model and run in sandboxes by the differential runner;
- a static tool's warning on the claimed lines that a verifier confirms is the same problem
  (tier 2);
- a verifier, told to refute the claim, that fails to (tier 3).
Candidates with none of these are tier 4, and the gate later drops them.
"""
