# ============================================================
# IoMT IDS Artifact (SBSeg 2026)
# Paper: Beyond Aggregate Accuracy: Kill Chain-Aware Evaluation
#        of IoMT Intrusion Detection Models
#
# CPU-only reproducible environment. A GPU is optional and not
# required for any experiment. Fixed seeds improve repeatability, but neural
# results may vary slightly across operating systems and CPU kernels.
#
# Build:
#   docker build -t iomt-killchain-artifact:sbseg2026 .
#
# Fast checks (no dataset needed):
#   docker run --rm iomt-killchain-artifact:sbseg2026            # make smoke
#   docker run --rm iomt-killchain-artifact:sbseg2026 make verify
#
# Reproduce claims (mount the CICIoMT2024 dataset into /app/data):
#   docker run --rm -v "$(pwd)/data:/app/data" \
#       -v "$(pwd)/results:/app/results" \
#       iomt-killchain-artifact:sbseg2026 make reproduce-autoencoder
# ============================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    MPLBACKEND=Agg

# System libraries: libgomp1 (OpenMP for LightGBM/TensorFlow), make (Makefile
# targets), git (optional provenance).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the artifact (source, pre-computed results, docs). The dataset is not
# included; mount it at /app/data for the dataset-dependent experiments.
COPY . .

# Default command: fast installation check with synthetic data (no dataset).
CMD ["make", "smoke"]
