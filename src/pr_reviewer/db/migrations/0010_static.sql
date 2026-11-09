-- Migration 0010: static tool results.
-- pr_static has one row per PR and tools setup: whether the tools ran, where the full result
-- is kept (a JSON file under data/static/, with every warning on the changed files), and the
-- counts results get summed by. static_warnings has one row per warning on a line the PR
-- added or changed: the ones the reviewer sees, and later tier-2 evidence.
-- How to run: pr-reviewer db migrate

CREATE TABLE pr_static (
    instance_id   TEXT NOT NULL REFERENCES bench_prs (instance_id),
    settings_key  TEXT NOT NULL,            -- fingerprint of the rules, code and tools image
    status        TEXT NOT NULL CHECK (status IN ('ok', 'no_python', 'failed')),
    head_commit   TEXT,                     -- where the files were checked; NULL when failed
    image         TEXT,                     -- the PR image the tools ran in
    path          TEXT,                     -- the result file, relative to the project
    changed_files INTEGER NOT NULL,
    warnings      INTEGER NOT NULL,         -- on the changed files
    kept          INTEGER NOT NULL,         -- on changed lines
    seconds       REAL NOT NULL,
    runs          JSONB NOT NULL,           -- each tool: status, seconds, mypy's mode, counts
    detail        TEXT NOT NULL,            -- why it failed; empty otherwise
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (instance_id, settings_key)
);

CREATE TABLE static_warnings (
    instance_id  TEXT NOT NULL,
    settings_key TEXT NOT NULL,
    tool         TEXT NOT NULL,             -- ruff, mypy, bandit or semgrep
    rule         TEXT NOT NULL,             -- such as 'F821' or 'attr-defined'
    file         TEXT NOT NULL,
    line         INTEGER NOT NULL,
    end_line     INTEGER NOT NULL,
    severity     TEXT NOT NULL,
    message      TEXT NOT NULL,
    FOREIGN KEY (instance_id, settings_key) REFERENCES pr_static (instance_id, settings_key)
        ON DELETE CASCADE
);

CREATE INDEX static_warnings_rule ON static_warnings (settings_key, tool, rule);
