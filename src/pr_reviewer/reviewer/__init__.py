"""The PR reviewer itself.

Will hold a LangGraph graph that gathers context for a PR, proposes candidate issues, collects
evidence for each one (a failing test, a static-tool warning or reasoning that survives a
verifier), and lets only well-backed findings through as comments.

Empty for now; filled in a later milestone. Imported, not run.
"""
