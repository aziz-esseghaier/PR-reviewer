"""The PR reviewer itself.

- budget.py: fit a PR's text into a model's request size limit, by whole file diffs.
- naive.py: the naive reviewer, a LangGraph graph that asks the model for findings from the
  diff alone and checks its JSON reply: the baseline.
- contextual.py: the context reviewer, the same graph with a retrieve step that adds code
  from the repo around the change (from context/).
- store.py: keep review runs, reviews and findings in PostgreSQL.
- report.py: write a run's results folder, with the reviews file `pr-reviewer score` reads.
- compare.py: compare two scored runs on the same PRs, overall, on subsets and PR by PR.
- commands.py: the `pr-reviewer review ...` commands.

Later milestones add candidates, evidence and the gate. Imported, not run.
"""
