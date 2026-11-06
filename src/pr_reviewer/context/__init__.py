"""Context retrieval, the RAG part: find the code around a PR's change.

- probe.py: the script that runs inside an indexer sandbox: it indexes the repo with `ast`
  (Jedi for what `ast` can't resolve) and collects candidate pieces: changed functions,
  definitions they call, callers, related tests and changed files.
- candidates.py: check the probe's answer and keep it as a JSON file per PR.
- retrieve.py: the indexer image, and running the probe in a sandbox.
- store.py: keep where each PR's candidates are, with their counts, in PostgreSQL.
- pack.py: fit whole pieces into the room a request has left, in a fixed order.
- check.py: the retrieval check: did the model see the code each known issue's fix changed?
- commands.py: the `pr-reviewer context ...` commands.

The reviewer that uses the context is reviewer/contextual.py. Imported, not run.
"""
