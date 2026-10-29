-- Migration 0003: the LLM usage log.
-- Creates the llm_usage table, with one row per request sent to a provider (every try,
-- failed ones included) and one per reply served from the cache, and the llm_usage_daily
-- view, which sums it per day and model. The client reads it to stay under daily limits
-- (see src/pr_reviewer/core/usage.py).
-- How to run: pr-reviewer db migrate

CREATE TABLE llm_usage (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Plain columns, because we count and group by all of them.
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    profile       TEXT NOT NULL,
    outcome       TEXT NOT NULL CHECK (outcome IN (
                      'ok', 'cache_hit', 'rate_limited', 'server_error',
                      'timeout', 'connection_error', 'rejected')),
    status_code   INTEGER,          -- NULL for cache hits, timeouts and dropped connections
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    latency_s     REAL              -- NULL for cache hits
);

-- The daily-limit check asks "this model's requests since time X" before every request.
CREATE INDEX llm_usage_model_time_idx ON llm_usage (provider, model, created_at);

-- Daily totals per model. Days are in UTC.
CREATE VIEW llm_usage_daily AS
SELECT
    (created_at AT TIME ZONE 'UTC')::date                                 AS day,
    provider,
    model,
    count(*) FILTER (WHERE outcome <> 'cache_hit')                        AS requests,
    count(*) FILTER (WHERE outcome NOT IN ('ok', 'cache_hit'))            AS failed,
    coalesce(sum(input_tokens + output_tokens)
             FILTER (WHERE outcome <> 'cache_hit'), 0)                    AS tokens,
    count(*) FILTER (WHERE outcome = 'cache_hit')                         AS cache_hits,
    coalesce(sum(input_tokens + output_tokens)
             FILTER (WHERE outcome = 'cache_hit'), 0)                     AS tokens_saved
FROM llm_usage
GROUP BY 1, 2, 3;
