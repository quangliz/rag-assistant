"""Vector store management module."""
import streamlit as st
from langchain_postgres import PGEngine, PGVectorStore
# from langchain_postgres.v2.indexes import HNSWIndex
from src.models import get_embedding
from src.config import PG_CONNECTION_STRING, VECTOR_TABLE_NAME, VECTOR_SIZE


@st.cache_resource
def get_vector_store():
    """
    Get cached vector store connection with automatic table initialization.
    
    The vector store is cached to avoid creating multiple connections.
    Uses pgvector extension for efficient similarity search.
    Handles both first-time initialization and subsequent calls gracefully.
    
    Returns:
        PGVectorStore: PostgreSQL vector store with pgvector extension
        
    Raises:
        ValueError: If connection string is invalid or missing
        ConnectionError: If unable to connect to PostgreSQL
    """
    if not PG_CONNECTION_STRING:
        raise ValueError("PG_CONNECTION_STRING environment variable is not set")
    
    try:
        # Create database engine
        engine = PGEngine.from_connection_string(url=PG_CONNECTION_STRING)
        
        # Try to initialize the vector store table
        # This will create the table if it doesn't exist
        # If it already exists, we catch and ignore the error
        try:
            engine.init_vectorstore_table(
                table_name=VECTOR_TABLE_NAME,
                vector_size=VECTOR_SIZE,
            )
        except Exception as init_error:
            # Table likely already exists, which is fine
            # We'll proceed to create the vector store instance
            error_msg = str(init_error).lower()
            if "already exists" not in error_msg and "duplicate" not in error_msg:
                # If it's not a "table exists" error, re-raise it
                raise
        
        # Create and return the vector store instance
        return PGVectorStore.create_sync(
            engine=engine,
            table_name=VECTOR_TABLE_NAME,
            embedding_service=get_embedding(),
        )
    except Exception as e:
        raise ConnectionError(f"Failed to initialize vector store: {str(e)}") from e



def add_documents_to_store(docs):
    """
    Add documents to the vector store.
    
    Args:
        docs: List of Document objects to add
        
    Returns:
        int: Number of documents added
    """
    vector_store = get_vector_store()
    vector_store.add_documents(docs)
    return len(docs)


def clear_vector_store():
    """
    Delete all documents from the vector store by dropping and recreating the table.
    
    Note: This will clear the Streamlit cache and reinitialize the vector store.
    """
    try:
        # Create database engine
        engine = PGEngine.from_connection_string(url=PG_CONNECTION_STRING)        
        # Drop the existing table
        try:
            engine.drop_table(table_name=VECTOR_TABLE_NAME)
        except Exception:
            # Table might not exist, which is fine
            pass
        
        # Recreate the table
        engine.init_vectorstore_table(
            table_name=VECTOR_TABLE_NAME,
            vector_size=VECTOR_SIZE,
        )
        
        # Clear the cached vector store so it reinitializes
        st.cache_resource.clear()
        
    except Exception as e:
        raise ConnectionError(f"Failed to clear vector store: {str(e)}") from e
