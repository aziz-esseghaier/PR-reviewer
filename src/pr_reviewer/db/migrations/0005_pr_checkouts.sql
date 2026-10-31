-- Migration 0005: where each benchmark PR's code can be checked out.
-- One row per PR checked by `pr-reviewer sandbox check`: its base (the commit just before its
-- first reviewed commit, found with git), its PR state (its last reviewed commit), whether
-- both sides can be checked out, and how the dataset's base_commit field relates to the base.
-- results/commit-check/ holds the same checks as files.
-- How to run: pr-reviewer db migrate

CREATE TABLE pr_checkouts (
    instance_id         TEXT PRIMARY KEY REFERENCES bench_prs (instance_id),
    -- All reviewed commits found, and both sides checked out and run.
    usable              BOOLEAN NOT NULL,
    base_sha            TEXT,               -- the commit just before the first reviewed commit
    pr_sha              TEXT NOT NULL,      -- the last reviewed commit: what the reviewer sees
    -- Where the dataset's base_commit field is, seen from the base: the same commit, later or
    -- earlier on the same line of history, on another line, or missing from the repo.
    base_field_relation TEXT CHECK (
        base_field_relation IN ('same', 'later', 'earlier', 'elsewhere', 'missing')
    ),
    merge_commits       INTEGER NOT NULL,   -- merge commits among the reviewed commits
    -- Base vs PR state changes only files the PR's own commits changed.
    only_pr_changes     BOOLEAN,
    problem             TEXT,               -- why the PR isn't usable, when it isn't
    details             JSONB NOT NULL,     -- the whole check, as in results/commit-check/raw/
    checked_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
