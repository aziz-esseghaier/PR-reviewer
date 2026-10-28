-- Turn on pgvector, which adds a vector column type and similarity search.
-- The RAG part (M6) may use it to store embeddings.
CREATE EXTENSION IF NOT EXISTS vector;
