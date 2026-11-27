FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    ORT_INTRA_THREADS=2 \
    ORT_INTER_THREADS=1 \
    STAGE1_MODEL_PATH=/app/models/woundvision_v6_stage1_wound.onnx \
    STAGE2_MODEL_PATH=/app/models/woundvision_v6_stage2_tissue.onnx \
    METRICS_ENABLED=true \
    ENABLE_JSON_LOGS=true \
    METRICS_DB_PATH=/app/telemetry/metrics.db

# Create non-root user with UID 1000
RUN useradd --create-home --uid 1000 --shell /bin/bash user

WORKDIR /app

# Install lightweight runtime dependencies (no PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models directory
COPY --chown=user:user inference.py monitoring.py app.py ./
COPY --chown=user:user models/ ./models/
RUN mkdir -p /app/telemetry && chown user:user /app/telemetry

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", \"7860\")}/health', timeout=5)"

# Single worker keeps RAM low (~515 MB peak for Two-Stage ONNX)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
