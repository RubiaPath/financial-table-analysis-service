#!/bin/bash
set -e

CONTAINER_NAME="table-analysis-test"
IMAGE_NAME="financial-table-analysis:test"
PORT=8080

echo "Local Docker Test"
echo

# Stop existing container
echo "Cleaning up old container..."
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
echo "Starting container with mounted models..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8080 \
    -v "$(pwd)/models:/opt/ml/model" \
    -v "$(pwd)/config.yaml:/opt/program/config.yaml" \
    $IMAGE_NAME

echo "Container started. Waiting for service..."
sleep 5

# Test health check
echo "Testing health endpoint..."
for i in {1..30}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Service is ready!"
        break
    fi
    echo "  Attempt $i/30..."
    sleep 2
done

# Show logs
echo
echo "=== Container logs ==="
docker logs $CONTAINER_NAME | tail -20

echo
echo "=== Health check ==="
curl -s http://localhost:$PORT/health | python3 -m json.tool

echo
echo "Container: $CONTAINER_NAME"
echo "API: http://localhost:$PORT"
echo "Docs: http://localhost:$PORT/docs"
echo
echo "Stop: docker stop $CONTAINER_NAME"
echo "Logs: docker logs -f $CONTAINER_NAME"
