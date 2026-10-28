"""Agent that builds a working environment for a benchmark repo.

Will hold a LangGraph graph that installs a repo's dependencies in Docker, checks the result
by running the repo's own tests, and saves what worked as a recipe, cached per repo and time
period, so the agent only runs when no recipe fits.

Empty for now; filled in a later milestone. Imported, not run.
"""
