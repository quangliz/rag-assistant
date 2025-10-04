"""Configuration constants for the RAG application."""
import os

# Model names
OPENAI_MODEL_NAME = "gpt-5-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
RERANKER_MODEL_NAME = "rerank-english-v3.5"

# PostgreSQL connection
PG_CONNECTION_STRING = os.getenv(
    "PG_CONNECTION_STRING",
    "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
)

# Vector store configuration
VECTOR_TABLE_NAME = "vectorstore"
VECTOR_SIZE = 1536  # openai embedding dimension

# Retrieval configuration
DEFAULT_TOP_N = 5  # Number of final documents to return
DEFAULT_INITIAL_K = 20  # Number of documents to retrieve before reranking

# Text splitting configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

