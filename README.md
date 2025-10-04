### Overview
A conversational RAG (Retrieval-Augmented Generation) chatbot that allows you to chat with your documents using advanced reranking for better relevance.

### Features
- **Conversational Interface**: Chat naturally with your documents, maintaining context across the conversation
- **Multiple Document Sources**: Upload PDFs, DOCX, PPTX, images, or fetch content from URLs
- **Advanced Reranking**: Uses Cohere's reranking API to improve search relevance (retrieves 20 docs, reranks to top 5)
- **Persistent Storage**: Documents stored in PostgreSQL with pgvector for efficient similarity search
- **Configurable Retrieval**: Adjust reranking parameters in real-time

### Setup

#### Option 1: Docker (Recommended)

1. **Clone this repo**
```bash
git clone https://github.com/quangliz/rag-assistant.git
cd rag-assistant
```

2. **Build and run with Docker**
```bash
# Build the Docker image
docker build -t rag-search:latest .

# Start the database
docker compose up -d

# Run the application (will be available at http://localhost:8501)
docker run -p 8501:8501 --network host rag-search:latest
```

#### Option 2: Manual Setup

1. **Clone this repo**
```bash
git clone https://github.com/quangliz/rag-assistant.git
cd rag-assistant
```

2. **Install dependencies**
```bash
# Install uv if not yet installed
pip install uv

# Install requirements
uv sync

# Set up environment variables (optional)
cp .env.example .env  # Add your API keys
```

3. **Start the database**
```bash
docker compose up -d
```

4. **Run Streamlit interface**
```bash
uv run streamlit run app.py
```

### How It Works

1. **Document Processing**: Upload documents or URLs → Documents are split into chunks → Embedded using OpenAI → Stored in PostgreSQL
2. **Retrieval with Reranking**:
   - Initial retrieval: Vector search fetches top 20 similar chunks
   - Reranking: Cohere reranks these 20 chunks to find the most relevant 5
   - Context: Top 5 chunks are used as context for the LLM
3. **Conversational Response**: LLM generates a response using the reranked context and chat history

### To-dos
- [x] Process uploaded documents
- [x] Support URLs
- [x] Conversational interface with chat history
- [x] Reranking
- [ ] Process web search results (Tavily)