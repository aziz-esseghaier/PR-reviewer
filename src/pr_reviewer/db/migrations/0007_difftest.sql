-- Migration 0007: differential test runs.
-- diff_runs has one row per differential run: the same tests run several times at two sides
-- of a PR (usually its base and its PR state), with how many tests got each outcome. The
-- details column holds every test's outcome, each run's status at each side, and the
-- failures' exception types and lines. request_key fingerprints everything that decides the
-- result (images, test code, targets, settings), so the same run is never done twice.
-- How to run: pr-reviewer db migrate

CREATE TABLE diff_runs (
    id            SERIAL PRIMARY KEY,
    request_key   TEXT NOT NULL UNIQUE,
    instance_id   TEXT NOT NULL REFERENCES bench_prs (instance_id),
    old_side      TEXT NOT NULL,      -- such as 'base'
    new_side      TEXT NOT NULL,      -- such as 'pr'
    runs_per_side INTEGER NOT NULL,
    tests         INTEGER NOT NULL,
    pass_fail     INTEGER NOT NULL,   -- passes before, fails after: the outcome that can prove a bug
    pass_pass     INTEGER NOT NULL,
    fail_pass     INTEGER NOT NULL,
    fail_fail     INTEGER NOT NULL,
    flaky         INTEGER NOT NULL,   -- runs disagreed at a side
    not_run       INTEGER NOT NULL,   -- timed out, out of memory, skipped or not collected
    minutes       REAL NOT NULL,
    details       JSONB NOT NULL,     -- the DiffResult: every test's outcome and failures
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX diff_runs_instance ON diff_runs (instance_id, created_at);
