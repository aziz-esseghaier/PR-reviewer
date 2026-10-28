"""Core building blocks shared by the rest of the project.

- config.py: loads .env and configs/models.yaml, and checks them.
- llm.py: the LLM client, the only route from our code to a model.
- cache.py: the PostgreSQL cache of model replies.

Rate limits and shared schemas (Finding, Evidence, EnvRecipe) will be added here.
Imported by other modules; nothing here is run directly.
"""
