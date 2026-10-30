-- Migration 0001: turn on pgvector.
-- pgvector adds a vector column type and similarity search to PostgreSQL. The context
-- retrieval (RAG) part may use it to store embeddings and find similar code.
-- How to run: pr-reviewer db migrate

CREATE EXTENSION IF NOT EXISTS vector;
