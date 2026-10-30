-- Migration 0004: benchmark PRs and scored experiments.
-- bench_prs lists the SWR-Bench PRs the benchmark uses, with what results get grouped by:
-- repo, year, kind and dev or test set. experiments has one row per scored set of reviews, and
-- pr_scores one row per PR of each, counted the way SWR-Bench's scorer counts them. The folder
-- results/<experiment>/ holds the same scores, plus the judge's full replies.
-- How to run: pr-reviewer db migrate

CREATE TABLE bench_prs (
    instance_id         TEXT PRIMARY KEY,
    -- Plain columns, because results are grouped and counted by all of them.
    repo                TEXT NOT NULL,
    year                INTEGER NOT NULL,
    kind                TEXT NOT NULL CHECK (kind IN ('functional', 'clean')),
    split               TEXT NOT NULL CHECK (split IN ('dev', 'test')),
    functional_issues   INTEGER NOT NULL,
    evolvability_issues INTEGER NOT NULL
);

CREATE TABLE experiments (
    name        TEXT PRIMARY KEY,
    split       TEXT NOT NULL CHECK (split IN ('dev', 'test')),
    reviewer    TEXT NOT NULL,          -- the reviewer's model, or 'none' for a baseline
    judge_model TEXT NOT NULL,
    scored_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    config      JSONB NOT NULL          -- the same as results/<name>/config.json
);

CREATE TABLE pr_scores (
    experiment   TEXT NOT NULL REFERENCES experiments (name) ON DELETE CASCADE,
    instance_id  TEXT NOT NULL REFERENCES bench_prs (instance_id),
    judged_good  BOOLEAN NOT NULL,      -- the review passed the PR: said so, or raised no point
    points       INTEGER NOT NULL,      -- review points the judge found
    hits         INTEGER NOT NULL,      -- points that matched a known issue
    false_alarms INTEGER NOT NULL,      -- points that matched no known issue
    misses       INTEGER NOT NULL,      -- known issues that no point matched
    judgment     JSONB NOT NULL,        -- the judge's points and matches, as SWR-Bench keeps them
    PRIMARY KEY (experiment, instance_id)
);
