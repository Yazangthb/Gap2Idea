FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip uv

# Copy only dependency manifests first (best caching)
COPY pyproject.toml uv.lock ./

# Create venv + install deps
RUN uv sync --frozen --no-dev

# Now copy the application code
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "src/gap2idea/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
