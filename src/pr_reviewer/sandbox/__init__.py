"""Runs untrusted PR code safely inside Docker containers.

Will hold the Docker runner (CPU, memory and time limits, network off while tests run), the
git checkout of a repo at a given commit, and the differential test runner: a test that passes
on the base commit and fails on the PR commit is the strongest evidence a finding can have.

Empty for now; filled in a later milestone. Imported, not run.
"""
