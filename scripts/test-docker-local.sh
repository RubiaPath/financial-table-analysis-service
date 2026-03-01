#!/bin/bash
set -e

CONTAINER_NAME="table-analysis-test"
IMAGE_NAME="financial-table-analysis:test"

echo "Local Docker Test"
echo

# Stop existing container
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Check if models exist
if [ ! -d "models/sam3/checkpoints" ] || [ ! -d "models/ollama/models" ]; then
    echo "ERROR: models not found. Run:"
    echo "  bash scripts/download_sam3_model.sh"
    echo "  bash scripts/download_ollama_models.sh"
    exit 1
fi

# Build Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Run container with mounted models
echo "Starting container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p 8080:8080 \
    -v "$(pwd)/models:/opt/ml/model" \
    -v "$(pwd)/config.yaml:/opt/program/config.yaml" \
    $IMAGE_NAME

# Show logs
echo
echo "=== Container logs (follow with Ctrl+C) ==="
docker logs -f $CONTAINER_NAME
