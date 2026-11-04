-- Migration 0006: environment recipes and setup runs.
-- env_recipes holds every recipe that built a working environment (the repo's own tests
-- passed at a PR's base and PR state), per repo and year, with the Docker image saved from
-- it. setup_runs has one row per setup of a PR: whether it worked, what got it working
-- (a cached recipe, the standard recipe or the agent), or why it failed. The logs stay in
-- files; log_dir points to them.
-- How to run: pr-reviewer db migrate

CREATE TABLE env_recipes (
    id          SERIAL PRIMARY KEY,
    repo        TEXT NOT NULL,
    year        INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,          -- what the recipe does (Python, steps, test command)
    source      TEXT NOT NULL CHECK (source IN ('standard', 'agent')),
    recipe      JSONB NOT NULL,         -- the EnvRecipe
    image       TEXT,                   -- the environment image built from it
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (repo, year, fingerprint)
);

CREATE TABLE setup_runs (
    id               SERIAL PRIMARY KEY,
    instance_id      TEXT NOT NULL REFERENCES bench_prs (instance_id),
    thread_id        TEXT NOT NULL,     -- the setup graph's saved progress
    repo             TEXT NOT NULL,
    year             INTEGER NOT NULL,
    verified         BOOLEAN NOT NULL,  -- the related tests passed at the base and PR state
    -- What got it working, or NULL when nothing did.
    source           TEXT CHECK (source IN ('cache', 'standard', 'agent')),
    failure_category TEXT,              -- why it failed, from the logs; NULL when verified
    recipe_id        INTEGER REFERENCES env_recipes (id),
    attempts         INTEGER NOT NULL,
    model_calls      INTEGER NOT NULL,
    minutes          REAL NOT NULL,
    base_pass_rate   REAL,
    pr_pass_rate     REAL,
    tests_run        INTEGER,           -- tests that ran at the base
    agent_model      TEXT NOT NULL,
    details          JSONB NOT NULL,    -- every attempt, with its recipe, outcome and log folder
    finished_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX setup_runs_instance ON setup_runs (instance_id, finished_at);
