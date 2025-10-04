from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import trafilatura
import tempfile
import os
from src.vector_store import get_vector_store
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def process_uploaded_files(files) -> list[Document]:
    """
    Process uploaded files
    Args:
        files: A list of Streamlit UploadedFile objects
    Returns:
        A list of Document
    """
    docs = []
    for file in files:
        # Create a temporary file with the same extension as the uploaded file
        suffix = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            # Write the uploaded file content to the temporary file
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Process the temporary file(allow processing images)
            loader = DoclingLoader(file_path=tmp_file_path, export_type=ExportType.MARKDOWN)
            docs_load = loader.load()
            # map source metadata to the original file name
            docs_load = [Document(page_content=doc.page_content, metadata={"source": file.name}) for doc in docs_load]
            docs.extend(docs_load)  # Use extend instead of append to flatten the list
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    return docs

def process_urls(url: str) -> list[Document]:
    """
    Process data from urls using trafilatura
    Args:
        url: A url
    Returns:
        A list of Document with metadata containing the url
    """
    docs = []
    try:
        downloaded = trafilatura.fetch_url(url)
        main_text = trafilatura.extract(downloaded, output_format="markdown")
        print("done")
        docs.append(Document(page_content=main_text, metadata={"source": url}))
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return []
    return docs

def split_docs(docs: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks.
    
    Args:
        docs: A list of Document objects
        
    Returns:
        A list of Document chunks with preserved metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = [
        Document(page_content=split, metadata=doc.metadata)
        for doc in docs
        for split in splitter.split_text(doc.page_content)
    ]
    return splits

def store_docs(docs: list[Document]) -> None:
    """
    Store documents in a vector store
    Args:
        docs: A list of Document
    """
    vector_store = get_vector_store()
    vector_store.add_documents(docs)
