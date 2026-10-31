"""Runs untrusted PR code safely inside Docker containers.

- config.py: the sandbox settings in configs/sandbox.yaml (limits, time limits, Docker names).
- image.py: builds the sandbox image from docker/sandbox/Dockerfile.
- runner.py: one sandbox container, with CPU, memory, process and time limits, no privileges,
  and a network switch that starts off.
- repos.py: one shared git copy of each repo in a Docker volume, read-only in sandboxes.
- checkout.py: puts a sandbox's working copy at a PR's base or PR state.
- commit_check.py: checks that every dev PR can be checked out on both sides, offline.
- commands.py: the `pr-reviewer sandbox ...` commands.

Still to come: the differential test runner, where a test that passes on the base and fails
on the PR state is the strongest evidence a finding can have.

Imported, not run.
"""
