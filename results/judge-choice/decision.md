# Judge model decision (2026-10-30)

**Judge: `gemini-3.5-flash-lite` (Google AI Studio free tier), fixed for the whole project.** Settings are SWR-Bench's scorer defaults: temperature 0, replies up to 8,192 tokens. Profile `judge-gemini-3.5-flash-lite` in `configs/models.yaml`, pinned in `src/pr_reviewer/benchmark/judge.py`.

## What the judge has to handle

SWR-Bench's scorer calls the judge once per PR. On the dev set (130 PRs), a judge prompt is about 2,300 tokens for a clean PR and 5,000 for a PR with issues (largest 13,000), about 430,000 prompt tokens per scoring run.

## Ruled out before the trial

- Groq models (gpt-oss-120b, gpt-oss-20b, gpt-oss-safeguard-20b, qwen3.8-27b): the free tier allows 8,000 tokens per minute (counting the 8,192-token reply allowance) and 200,000 per day, so no judge request fits, and a scoring run would take about three days.
- gemini-2.5-flash-lite: retired for new users.
- gemma-4-31b-it: server errors whenever a JSON format is requested.
- Gemini Flash (non-Lite) models: 20 requests a day on the free tier.

## Trial

`scripts/compare_judges.py` ran SWR-Bench's own per-PR scoring on 24 test reviews with known right answers, on 7 dev PRs (see `summary.md`). All three candidates (gemini-3.1-flash-lite, gemma-4-26b-a4b-it, gemini-3.5-flash-lite) scored all 24 correctly, so the trial didn't separate them on quality. gemma-4-26b-a4b-it had many server errors and rate limits and was the slowest.

## Why gemini-3.5-flash-lite

- The newest of the three, so the least likely to be retired: a judge that disappears would break comparisons across the project.
- Fast and reliable in the trial, with a known free quota of 500 requests a day (about three dev scoring runs).
- It stops being a reviewer model, so no model grades its own reviews.
