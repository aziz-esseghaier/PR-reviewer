"""The setup agent: builds a working environment for a benchmark PR, checked by its own tests.

- probe.py, facts.py: read a repo's setup files inside a sandbox (Python versions, CI, extras).
- versions.py: pick a Python version for a repo at a past date.
- selection.py: pick the existing tests related to a PR.
- standard.py: the usual install recipe, tried before asking the model.
- install.py: install a recipe in a sandbox, with packages pinned to the PR's date.
- testrun.py: run the related tests with the network off, and read the counts.
- attempt.py: try one recipe on a PR, at its base and PR state, in a fresh sandbox.
- store.py: the recipe cache and the setup runs, in PostgreSQL.
- agent.py: the LangGraph graph: cache, standard recipe, then the model, within a budget.
- report.py: success rates by repo and year, failure causes, and the usable subset.
- commands.py: the `pr-reviewer setup ...` commands.

Imported, not run.
"""
