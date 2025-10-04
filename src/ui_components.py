"""Reusable Streamlit UI components."""
import os
import time
import streamlit as st
from src.process_data import process_uploaded_files, process_urls, split_docs
from src.vector_store import add_documents_to_store, clear_vector_store


def _initialize_session_state():
    """Initialize session state variables if not already set."""
    if "processed_sources" not in st.session_state:
        st.session_state.processed_sources = set()
    if "OPENAI_API_KEY" not in st.session_state:
        st.session_state.OPENAI_API_KEY = None
    if "COHERE_API_KEY" not in st.session_state:
        st.session_state.COHERE_API_KEY = None


def _show_success_message(message: str, duration: int = 5):
    """Display a temporary success message that auto-dismisses."""
    placeholder = st.empty()
    placeholder.success(message)
    time.sleep(duration)
    placeholder.empty()


def _get_api_key_status(key_name: str, session_key: str) -> tuple[str, str]:
    """
    Get the status of an API key.
    
    Args:
        key_name: Environment variable name
        session_key: Session state key name
        
    Returns:
        Tuple of (status_type, status_message)
        status_type: 'success', 'info', 'error', or 'warning'
    """
    if st.session_state.get(session_key):
        return ("success", "✅ Session")
    elif os.getenv(key_name):
        return ("info", "ℹ️ .env")
    elif key_name == "COHERE_API_KEY":
        return ("warning", "⚠️ Not set")
    else:
        return ("error", "❌ Not set")


def render_settings_panel():
    """
    Render the retrieval settings panel in the sidebar.
    
    Returns:
        dict: Settings with keys 'use_reranking', 'top_n', 'initial_k'
    """
    _initialize_session_state()
    st.header("⚙️ Settings")
    
    # API Configuration (at top for visibility)
    with st.expander("🔑 API Keys (Session Only)", expanded=False):
        st.caption("Enter API keys for this session only. Leave empty to use environment variables.")
        
        # OpenAI API Key input
        openai_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Required for LLM and embeddings. Leave empty to use .env file.",
            key="openai_key_input"
        )
        
        # Only set in session state if user actually entered something
        if openai_key_input:
            st.session_state.OPENAI_API_KEY = openai_key_input
        
        # Cohere API Key input (for reranking)
        cohere_key_input = st.text_input(
            "Cohere API Key",
            type="password",
            placeholder="Your Cohere key...",
            help="Optional. Required only for reranking. Leave empty to use .env file.",
            key="cohere_key_input"
        )
        
        # Only set in session state if user actually entered something
        if cohere_key_input:
            st.session_state.COHERE_API_KEY = cohere_key_input
        
        # Show current key status
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            status_type, status_msg = _get_api_key_status("OPENAI_API_KEY", "OPENAI_API_KEY")
            status_func = getattr(st, status_type)
            status_func(f"OpenAI: {status_msg}")
        
        with col2:
            status_type, status_msg = _get_api_key_status("COHERE_API_KEY", "COHERE_API_KEY")
            status_func = getattr(st, status_type)
            status_func(f"Cohere: {status_msg}")

    # Retrieval Settings
    with st.expander("🔍 Retrieval Settings", expanded=False):
        use_reranking = st.checkbox(
            "Enable Reranking",
            value=False,
            help="Rerank results using Cohere for better relevance"
        )
        
        if use_reranking and (st.session_state.COHERE_API_KEY or os.getenv("COHERE_API_KEY")):
            top_n = st.slider(
                "Final results",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of documents to return after reranking"
            )
            initial_k = st.slider(
                "Initial retrieval",
                min_value=10,
                max_value=100,
                value=20,
                help="Number of documents to retrieve before reranking"
            )
        else:
            top_n = st.slider(
                "Results to retrieve",
                min_value=3,
                max_value=20,
                value=5
            )
            initial_k = top_n
    
    return {
        "use_reranking": use_reranking,
        "top_n": top_n,
        "initial_k": initial_k
    }


def render_document_management():
    """Render the document upload and management panel in the sidebar."""
    _initialize_session_state()
    st.header("📁 Manage Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "pptx", "md", "png", "jpeg", "jpg", "webp"],
        accept_multiple_files=True,
    )
    
    # URL input
    url_input = st.text_input("Enter URL")
    
    # Process and store uploaded files
    if uploaded_files and st.button("Process & Store Files"):
        with st.spinner("Processing files..."):
            docs = process_uploaded_files(uploaded_files)
            docs = split_docs(docs)
            num_docs = add_documents_to_store(docs)
            
            # Track processed sources
            file_sources = {doc.metadata.get("source", "Unknown") for doc in docs}
            st.session_state.processed_sources.update(file_sources)
            
            _show_success_message(f"✅ Stored {num_docs} document chunks!")

    # Process and store URL
    if url_input and st.button("Process & Store URL"):
        with st.spinner("Fetching and processing URL..."):
            docs = process_urls(url_input)
            if docs:
                docs = split_docs(docs)
                num_docs = add_documents_to_store(docs)
                
                # Track processed sources
                url_sources = {doc.metadata.get("source", "Unknown") for doc in docs}
                st.session_state.processed_sources.update(url_sources)
                
                _show_success_message(f"✅ Stored {num_docs} document chunks from URL!")
            else:
                st.error("Failed to process URL")
    
    # Display processed files and URLs
    if st.session_state.processed_sources:
        with st.expander("📄 Processed Sources", expanded=False):
            for idx, source in enumerate(sorted(st.session_state.processed_sources), 1):
                st.text(f"{idx}. {source}")
    
    with st.expander("🛠️ Utilities", expanded=False):
        # Option to clear database
        if st.button("🗑️ Clear All Documents"):
            clear_vector_store()
            st.session_state.processed_sources.clear()
            st.warning("All documents deleted!")
        
        # Clear chat history button
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


def render_chat_history():
    """Display all chat messages from session state."""
    if "messages" not in st.session_state:
        return
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                display_sources(message["sources"])


def display_sources(sources):
    """
    Display sources in an expander.

    Args:
        sources: List of source dicts with 'source' and 'content' keys
    """
    if not sources:
        return
    
    with st.expander("📚 View Sources"):
        for i, source in enumerate(sources, 1):
            source_name = source.get('source', 'Unknown')
            content = source.get('content', '')
            
            st.caption(f"**Source {i}:** {source_name}")
            # Show preview (max 300 chars)
            preview = content[:300]
            st.text(preview + ("..." if len(content) > 300 else ""))
            
            if i < len(sources):  # Don't add divider after last source
                st.divider()
