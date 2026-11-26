FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    ORT_INTRA_THREADS=2 \
    ORT_INTER_THREADS=1 \
    MODEL_PATH=/app/models/woundvision_v5_augmented.onnx

# Create non-root user with UID 1000 (required by Hugging Face Spaces)
RUN useradd --create-home --uid 1000 --shell /bin/bash user

WORKDIR /app

# Install lightweight runtime dependencies (no PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models directory (bundles .onnx when present, e.g. HF Spaces)
COPY --chown=user:user inference.py app.py ./
COPY --chown=user:user models/ ./models/

# Optional: bake ONNX model at build time via --build-arg MODEL_URL=... (for GitHub/VPS deploys)
ARG MODEL_URL=""
RUN if [ -n "$MODEL_URL" ] && [ ! -f /app/models/woundvision_v5_augmented.onnx ]; then \
      MODEL_URL="$MODEL_URL" python -c "import os, urllib.request; urllib.request.urlretrieve(os.environ['MODEL_URL'], '/app/models/woundvision_v5_augmented.onnx')" && \
      chown user:user /app/models/woundvision_v5_augmented.onnx; \
    fi

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", \"7860\")}/health', timeout=5)"

# Single worker keeps RAM low (~270 MB); ONNX Runtime handles intra-op threading internally
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
