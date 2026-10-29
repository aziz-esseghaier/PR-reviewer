"""Core building blocks shared by the rest of the project.

- config.py: loads .env and configs/models.yaml, and checks them.
- llm.py: the LLM client, the only route from our code to a model.
- cache.py: the PostgreSQL cache of model replies.
- limits.py: retries, the per-minute throttle and the daily-limit math.
- usage.py: the PostgreSQL log of every request, used for daily limits and reports.
- schemas.py: shared data shapes: findings with their evidence, environment recipes, and the
  review text SWR-Bench scores.

Imported by other modules; nothing here is run directly.
"""
