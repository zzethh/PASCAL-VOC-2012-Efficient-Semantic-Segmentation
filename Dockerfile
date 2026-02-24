# ──────────────────────────────────────────────────────────────
# Dockerfile: PASCAL VOC Semantic Segmentation
# Builds training + inference environment in a single container
# ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

# Copy source code
COPY *.py ./

# Configurable via environment
ENV GPU_ID=0
ENV EPOCHS=50
ENV BATCH_SIZE=16
ENV GROUP_NUMBER=26
ENV DATA_ROOT=/app/data

# Default: run training
CMD ["sh", "-c", "python train.py \
    --data_root $DATA_ROOT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gpu $GPU_ID \
    --download"]
