import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_cohere import CohereRerank
import streamlit as st
from src.config import OPENAI_MODEL_NAME, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME

load_dotenv()


def get_api_key(key_name: str) -> str:
    """
    Get API key from session state if provided, otherwise from environment.
    
    Args:
        key_name: Name of the API key (e.g., 'OPENAI_API_KEY')
        
    Returns:
        API key string
        
    Raises:
        ValueError: If API key is not found in session or environment
    """
    # First check session state (user-provided)
    if key_name in st.session_state and st.session_state[key_name]:
        return st.session_state[key_name]
    
    # Fall back to environment variable
    env_key = os.getenv(key_name)
    if env_key:
        return env_key
    
    raise ValueError(f"{key_name} not found in session or environment variables")


def get_llm():
    """
    Get ChatOpenAI instance with session or environment API key.
    
    Returns:
        ChatOpenAI instance
    """
    api_key = get_api_key("OPENAI_API_KEY")
    return ChatOpenAI(model=OPENAI_MODEL_NAME, api_key=api_key)


def get_embedding():
    """
    Get OpenAI embeddings instance with session or environment API key.
    
    Returns:
        OpenAIEmbeddings instance
    """
    api_key = get_api_key("OPENAI_API_KEY")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=api_key)


# def get_reranker():
#     """
#     Get Cohere reranker instance with session or environment API key.
#     """
#     api_key = get_api_key("COHERE_API_KEY")
#     return CohereRerank(model=RERANKER_MODEL_NAME, api_key=api_key)