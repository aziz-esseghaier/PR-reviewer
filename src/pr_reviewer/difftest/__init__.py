"""The differential test runner: the same tests before and after a PR, compared.

Executed proof, the strongest evidence a reviewer finding can have, is a test that passes at
a PR's base and fails at its PR state. This package runs tests at both sides and says, per
test, which of these happened: pass_fail, pass_pass, fail_pass, fail_fail, or flaky (runs
disagree) or not_run (timeout, out of memory, skipped).

- environments.py: prepared images of each PR at its base and PR state.
- pytest_probe.py: a pytest plugin, copied into sandboxes, that writes down each test's result.
- reports.py: read one run's results (the plugin's, or Django's runner output).
- outcomes.py: combine the runs at each side, then the two sides, into one outcome per test.
- targets.py: find the tests a diff added or changed.
- runner.py: run tests at two sides, several times each, in sandboxes.
- store.py: keep runs in PostgreSQL and reuse them.
- validation.py: check the runner on real tests whose outcome is known.
- commands.py: the `pr-reviewer difftest ...` commands.

Imported, not run.
"""
