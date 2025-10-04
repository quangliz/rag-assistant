"""Main Streamlit application - UI orchestration only."""
import streamlit as st
from dotenv import load_dotenv

from src.vector_store import get_vector_store
from src.retrieval import get_retriever
from src.chat import generate_response
from src.ui_components import (
    render_settings_panel,
    render_document_management,
    render_chat_history,
    display_sources
)

# Load environment variables
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize API keys in session state (None means use environment variables)
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = None

if "COHERE_API_KEY" not in st.session_state:
    st.session_state.COHERE_API_KEY = None

# Page configuration
st.title("💬 Chat with Your Documents")

# Sidebar
with st.sidebar:
    # Render settings panel and get configuration
    settings = render_settings_panel()
    
    st.divider()
    
    # Render document management panel
    render_document_management()

# Display chat history
render_chat_history()

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get vector store
                vector_store = get_vector_store()
                
                # Get retriever with optional reranking
                retriever = get_retriever(
                    vector_store,
                    use_reranking=settings["use_reranking"],
                    top_n=settings["top_n"],
                    initial_k=settings["initial_k"]
                )
                
                # Retrieve relevant documents
                relevant_docs = retriever.invoke(prompt)
                
                # Generate response
                response, sources = generate_response(
                    prompt,
                    relevant_docs,
                    st.session_state.messages
                )
                
                # Display response
                st.markdown(response)
                
                # Show sources
                display_sources(sources)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
