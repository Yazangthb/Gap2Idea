FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip uv

# Copy dependency manifests + readme (referenced by pyproject) first for caching.
COPY pyproject.toml uv.lock README.md ./

# Install dependencies WITHOUT trying to build the project itself
# (src/ doesn't exist yet at this layer). This is the canonical uv pattern.
RUN uv sync --frozen --no-dev --no-install-project

# Cloud Run has no GPU, so strip CUDA wheels (~4.5 GB) and replace the
# default torch wheel with the CPU-only build (~200 MB instead of ~900 MB).
# Image drops from ~6 GB to ~1.5 GB. uv-managed venvs have no `pip`, so we
# use `uv pip` which operates on the active venv via UV_PROJECT_ENVIRONMENT.
RUN uv pip uninstall \
        torch triton cuda-bindings \
        nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
        nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
        nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
        nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
        nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 \
        || true \
    && uv pip install --no-cache \
        --index-url https://download.pytorch.org/whl/cpu \
        torch

# Now copy the application code and install ONLY the gap2idea package itself
# (no dep re-resolution, which would undo the CPU-torch swap above).
COPY . .
RUN uv pip install --no-cache --no-deps -e .

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "streamlit run src/gap2idea/app/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]
