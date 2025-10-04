"""Chat and conversation logic module."""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

from src.models import get_llm
from src.retrieval import format_docs_for_context


SYSTEM_PROMPT = """You are a helpful AI assistant having a conversation about documents. 
Use the provided context to answer questions accurately and naturally.
If you reference information from the context, you can mention it conversationally.
CRUCIAL: If the context doesn't contain relevant information, say "I don't have any information about that" and don't provide any additional information.

Context from documents:
{context}
"""


def convert_messages_to_langchain(messages):
    """
    Convert Streamlit session messages to LangChain message format.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        
    Returns:
        List of LangChain Message objects
    """
    chat_history = []
    for msg in messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    return chat_history


def create_conversational_chain(llm, relevant_docs, chat_history):
    """
    Create a conversational RAG chain.
    
    Args:
        llm: Language model instance
        relevant_docs: List of retrieved documents
        chat_history: List of previous messages (LangChain format)
        
    Returns:
        Runnable chain for generating responses
    """
    conversational_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    chain = (
        {
            "context": lambda x: format_docs_for_context(relevant_docs),
            "question": RunnablePassthrough(),
            "chat_history": lambda x: chat_history
        }
        | conversational_prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def generate_response(prompt, relevant_docs, session_messages):
    """
    Generate a conversational response based on retrieved documents.
    
    Args:
        prompt: User's current question
        relevant_docs: Retrieved documents from vector store
        session_messages: Full session message history
        
    Returns:
        tuple: (response_text, sources_list)
    """
    if not relevant_docs:
        return "I don't have any documents to answer your question. Please upload some documents first!", []
    
    # get llm
    llm = get_llm()
    
    # convert chat history (excluding current message)
    chat_history = convert_messages_to_langchain(session_messages[:-1])
    
    # create and invoke chain
    chain = create_conversational_chain(llm, relevant_docs, chat_history)
    response = chain.invoke(prompt)
    
    # prepare sources for display
    sources = [
        {
            "source": doc.metadata.get('source', 'Unknown'),
            "content": doc.page_content
        }
        for doc in relevant_docs
    ]
    
    return response, sources

