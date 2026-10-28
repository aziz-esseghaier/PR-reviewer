-- Migration 0002: the LLM response cache.
-- Creates the llm_cache table. Each row holds one model reply, stored under a fingerprint
-- of the request that produced it. Asking the same thing again is answered from this table
-- instead of calling the provider (see src/pr_reviewer/core/cache.py).
-- How to run: pr-reviewer db migrate

CREATE TABLE llm_cache (
    -- SHA-256 fingerprint of the request: provider, model, messages, settings and sample number.
    cache_key  TEXT PRIMARY KEY,
    -- Plain columns, so we can count and group replies by provider and model.
    provider   TEXT NOT NULL,
    model      TEXT NOT NULL,
    -- The full request and reply as JSON, so any cached answer can be inspected later.
    request    JSONB NOT NULL,
    response   JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
