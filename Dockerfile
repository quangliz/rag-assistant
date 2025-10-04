FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address", "0.0.0.0"]