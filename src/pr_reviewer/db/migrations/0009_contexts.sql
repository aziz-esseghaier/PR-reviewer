-- Migration 0009: retrieved context.
-- pr_contexts has one row per PR and retrieval setup: where the candidate pieces the probe
-- found are kept (a JSON file under data/contexts/), with the counts results get summed by as
-- columns and the probe's other stats as JSON. pr_reviews gets two columns saying how much of
-- that context a review's request held (NULL for reviewers without context, such as naive).
-- How to run: pr-reviewer db migrate

CREATE TABLE pr_contexts (
    instance_id       TEXT NOT NULL REFERENCES bench_prs (instance_id),
    settings_key      TEXT NOT NULL,        -- fingerprint of settings, probe code and image
    probe_version     INTEGER NOT NULL,
    head_commit       TEXT NOT NULL,        -- where the code was read
    path              TEXT NOT NULL,        -- the candidates file, relative to the project
    changed_functions INTEGER NOT NULL,
    pieces            INTEGER NOT NULL,
    seconds           REAL NOT NULL,
    stats             JSONB NOT NULL,       -- calls resolved by ast, Jedi or not at all, ...
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (instance_id, settings_key)
);

ALTER TABLE pr_reviews
    ADD COLUMN context_chars  INTEGER,      -- characters of context in the request
    ADD COLUMN context_pieces INTEGER;      -- pieces of context in the request
