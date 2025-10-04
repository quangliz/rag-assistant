"""Retrieval and reranking module."""
import os
import streamlit as st
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

def get_retriever(vector_store, use_reranking=True, top_n=5, initial_k=20):
    """
    Get a retriever with optional Cohere reranking for better relevance.
    
    Args:
        vector_store: The vector store to retrieve from
        use_reranking: Whether to use reranking (default: True)
        top_n: Number of top documents to return after reranking
        initial_k: Number of documents to retrieve before reranking
        
    Returns:
        Retriever instance (either standard or with reranking)
    """
    if not use_reranking:
        return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_n})
    
    return get_retriever_with_reranking(vector_store, top_n, initial_k)


def get_retriever_with_reranking(vector_store, top_n=5, initial_k=20):
    """
    Get a retriever with Cohere reranking for better relevance.
    
    Args:
        vector_store: The vector store to retrieve from
        top_n: Number of top documents to return after reranking
        initial_k: Number of documents to retrieve before reranking
        
    Returns:
        ContextualCompressionRetriever with Cohere reranking
    """
    # get base retriever with more documents
    base_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": initial_k})
    
    # check for Cohere API key (session state first, then environment)
    cohere_api_key = None
    if "COHERE_API_KEY" in st.session_state and st.session_state.COHERE_API_KEY:
        cohere_api_key = st.session_state.COHERE_API_KEY
    else:
        cohere_api_key = os.getenv("COHERE_API_KEY")
    
    if not cohere_api_key:
        st.warning("⚠️ COHERE_API_KEY not set. Reranking disabled. Add it in settings or .env file.")
        return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_n})
    
    # create reranker with session or environment API key
    compressor = CohereRerank(
        model="rerank-english-v3.5",
        top_n=top_n,
        cohere_api_key=cohere_api_key
    )
    
    # create compression retriever with reranking
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever

def format_docs_for_context(docs):
    """
    Format retrieved documents for LLM context.
    
    Args:
        docs: List of Document objects
        
    Returns:
        str: Formatted string with all documents
    """
    return "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    ])

