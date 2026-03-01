# SageMaker-compatible Docker image for Table Analysis Service
# Single endpoint with Ollama + SAM3 + FastAPI
# Model weights expected at /opt/ml/model/
#   - /opt/ml/model/sam3/checkpoints/...
#   - /opt/ml/model/ollama/models/...

# Base image (PyTorch + CUDA runtime)
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

LABEL maintainer="Financial Table Analysis Service"
LABEL description="SageMaker-compatible API for financial document table detection"

# Common runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    SM_MODEL_DIR=/opt/ml/model \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OLLAMA_MODELS=/opt/ml/model/ollama/models \
    SAM3_CHECKPOINT_DIR=/opt/ml/model/sam3/checkpoints

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl wget \
    && rm -rf /var/lib/apt/lists/*

# Working directory for application code
WORKDIR /opt/program

# Pin SAM3 source to a specific commit for reproducible builds
ARG SAM3_COMMIT=f6e51f59500a87c576c2df2323ce56b9fd7a12de

# Clone SAM3 and checkout the pinned commit
RUN git clone --filter=blob:none https://github.com/facebookresearch/sam3.git /opt/program/sam3 \
 && cd /opt/program/sam3 \
 && git checkout ${SAM3_COMMIT}

# Install Python dependencies
COPY requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir -r /opt/program/requirements.txt

# Install SAM3 from source at the pinned commit (editable install)
RUN pip install --no-cache-dir -e /opt/program/sam3

# Install Ollama (download or setup)
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy FastAPI application code
COPY src/ /opt/program/src/

# Add the SageMaker-compatible entrypoint script
COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

# Create model directories (will be mounted/populated at runtime)
RUN mkdir -p /opt/ml/model/sam3/checkpoints && \
    mkdir -p /opt/ml/model/ollama/models

# SageMaker inference containers listen on port 8080 by convention
EXPOSE 8080

# Start the server via the `serve` command (SageMaker standard)
CMD ["serve"]
