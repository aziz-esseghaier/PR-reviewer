# SWR-Bench official scripts

Copied unchanged from [ZZR0/SWRench](https://github.com/ZZR0/SWRench) at commit `67ae1d4395ac05f800d62b0e13678eadeb9fa5c7`. SWR-Bench is MIT licensed (see `LICENSE` in this folder).

- `evaluation_struct.py`: the official scoring script. An LLM judge splits a review into points and matches each one to the PR's known issues. The script then computes precision, recall and PR-level accuracy.
- `generation.py`: the official review-generation script. It shows how SWR-Bench builds the review prompt from a PR.
- `utils.py`: SWR-Bench's helper that calls the model. It can't be imported as is (line 13 reads the API key with `os.getenv[...]`, which raises a `TypeError`), and it has no response cache, so our scoring code will supply its own `run_chat`.

Don't edit these files. Their SHA-256 fingerprints are pinned in `src/pr_reviewer/benchmark/source.py`, and a test fails if they change. Ruff skips this folder, so formatting can't touch them.
