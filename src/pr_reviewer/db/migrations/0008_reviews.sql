-- Migration 0008: review runs and their findings.
-- review_runs has one row per run of a reviewer over a set of PRs, with its settings.
-- pr_reviews has one row per PR of a run: whether a review came out, the review text SWR-Bench
-- scores, and what it cost. findings has one row per finding, with the claim's fields as
-- columns (results get counted by category, file or confidence) and the full claim and its
-- evidence as JSON. Scores go to experiments and pr_scores (migration 0004), under the same name.
-- How to run: pr-reviewer db migrate

CREATE TABLE review_runs (
    name        TEXT PRIMARY KEY,           -- also the results/<name>/ folder
    reviewer    TEXT NOT NULL,              -- which reviewer, such as 'naive'
    profile     TEXT NOT NULL,              -- the model profile
    model       TEXT NOT NULL,
    split       TEXT NOT NULL CHECK (split IN ('dev', 'test')),
    config      JSONB NOT NULL,             -- every setting, as in review_config.json
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE pr_reviews (
    run           TEXT NOT NULL REFERENCES review_runs (name) ON DELETE CASCADE,
    instance_id   TEXT NOT NULL REFERENCES bench_prs (instance_id),
    status        TEXT NOT NULL CHECK (status IN ('done', 'failed')),
    failures      INTEGER NOT NULL,         -- reviews of this PR that failed, this one included
    tries         INTEGER NOT NULL,         -- replies asked for in the last review
    findings      INTEGER NOT NULL,
    trimmed       BOOLEAN NOT NULL,         -- the model saw less than the whole PR
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    review        TEXT NOT NULL,            -- the text SWR-Bench scores; empty when failed
    details       JSONB NOT NULL,           -- left-out files and the graph's log
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run, instance_id)
);

CREATE TABLE findings (
    run              TEXT NOT NULL,
    instance_id      TEXT NOT NULL,
    finding_id       TEXT NOT NULL,         -- f1, f2, ... in the model's order
    file             TEXT NOT NULL,
    start_line       INTEGER NOT NULL,
    end_line         INTEGER NOT NULL,
    category         TEXT NOT NULL,         -- SWR-Bench category, such as 'F.2'
    functional       BOOLEAN NOT NULL,      -- an F category
    confidence       REAL NOT NULL,
    expected_failure TEXT,                  -- the exception a test would hit, if any
    tier             INTEGER NOT NULL,      -- strongest evidence: 1 executed ... 4 none
    claim            JSONB NOT NULL,
    evidence         JSONB NOT NULL,
    PRIMARY KEY (run, instance_id, finding_id),
    FOREIGN KEY (run, instance_id) REFERENCES pr_reviews (run, instance_id) ON DELETE CASCADE
);

CREATE INDEX findings_category ON findings (run, category);
