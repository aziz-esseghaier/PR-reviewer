-- Migration 0011: evidence for candidate findings.
-- evidence_runs has one row per run of the evidence agent over a review run's candidates, with
-- its settings. candidate_evidence has one row per candidate: the tier it reached and how
-- (columns, so results can be counted by tier, proof kind or verdict with SQL), what it cost,
-- every piece of evidence as JSON, and the path of the generated test that proves it, when one
-- does. The test files themselves live under data/evidence/.
-- How to run: pr-reviewer db migrate

CREATE TABLE evidence_runs (
    name        TEXT PRIMARY KEY,           -- also the results/<name>/ folder
    source_run  TEXT NOT NULL REFERENCES review_runs (name),  -- where the candidates come from
    profile     TEXT NOT NULL,              -- the model profile
    model       TEXT NOT NULL,
    config      JSONB NOT NULL,             -- every setting, as in evidence_config.json
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE candidate_evidence (
    run              TEXT NOT NULL REFERENCES evidence_runs (name) ON DELETE CASCADE,
    instance_id      TEXT NOT NULL REFERENCES bench_prs (instance_id),
    finding_id       TEXT NOT NULL,         -- the candidate's ID in the source run, such as f3
    status           TEXT NOT NULL CHECK (status IN ('done', 'failed')),
    failures         INTEGER NOT NULL,      -- times this candidate failed, this one included
    functional       BOOLEAN NOT NULL,      -- an F category
    expected_failure TEXT,                  -- the exception a test would hit, if any
    tier             INTEGER NOT NULL,      -- strongest evidence: 1 executed ... 4 none
    proof            TEXT,                  -- 'executed' or 'new_code' for a tier-1 proof
    test_result      TEXT NOT NULL,         -- how the test writer ended, such as 'proof' or
                                            -- 'no_proof'; 'not_tried' when no test was written
    verdict          TEXT,                  -- the verifier's: 'upheld', 'refuted' or
                                            -- 'deliberate'; NULL when it wasn't asked
    tool_confirmed   BOOLEAN NOT NULL,      -- a static warning confirmed as the same problem
    test_path        TEXT,                  -- the proving test's file, relative to the project
    calls            INTEGER NOT NULL,      -- model replies asked for
    input_tokens     INTEGER NOT NULL,
    output_tokens    INTEGER NOT NULL,
    docker_minutes   REAL NOT NULL,         -- time spent running tests (stored runs count 0)
    evidence         JSONB NOT NULL,        -- every piece of evidence, supporting or not
    details          JSONB NOT NULL,        -- each attempt, the matcher's reasons, the log
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run, instance_id, finding_id)
);

CREATE INDEX candidate_evidence_tier ON candidate_evidence (run, tier);
