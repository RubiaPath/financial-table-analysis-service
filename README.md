# Financial Table Analysis Service

FastAPI microservice for extracting and analyzing financial tables from PDF documents using SAM3 (table detection) and Ollama LLM (classification).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
export OLLAMA_MODELS=/path/to/ollama/models
export SAM3_CHECKPOINT_DIR=/path/to/sam3/checkpoints

# Run API
python -m uvicorn src.main:app --host 0.0.0.0 --port 8080
```

## API Endpoints

- `GET /health` - Health check
- `POST /api/v1/analyze-page` - Analyze page (base64 image)
- `POST /api/v1/analyze-page-file` - Analyze page (file upload)

## SageMaker Deployment

```bash
cd scripts/
bash deploy.sh
```

See [docs/SAGEMAKER_DEPLOYMENT.md](docs/SAGEMAKER_DEPLOYMENT.md) for details.

## Architecture

- **SAM3**: Table detection via segment anything model
- **Ollama**: LLM-based classification (Qwen3-8B)
- **FastAPI**: HTTP API server

## Documentation

- [docs/README.md](docs/README.md) - Full documentation
- [docs/QUICKSTART.md](docs/QUICKSTART.md) - Quick start guide
- [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) - Architecture overview
- [scripts/README.md](scripts/README.md) - Deployment scripts

## Configuration

Environment variables:

- `API_HOST` - API host (default: 0.0.0.0)
- `API_PORT` - API port (default: 8080)
- `OLLAMA_HOST` - Ollama API host (default: http://127.0.0.1:11434)
- `OLLAMA_MODELS` - Path to Ollama models directory
- `SAM3_CHECKPOINT_DIR` - Path to SAM3 model directory

## Requirements

- Python 3.10+
- PyTorch 2.5.1+ with CUDA 12.1
- Ollama (optional for local inference)
- 8GB+ GPU VRAM recommended
