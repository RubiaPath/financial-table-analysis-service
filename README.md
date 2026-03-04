# Financial Table Analysis Service

FastAPI microservice for automated page classification and table detection from PDF documents using SAM3 (table detection) and Ollama LLM (text-based classification). Provides API service for the Balance Sheet Extraction Agent.

## Key Features

- **SAM3 Table Detection**: Segment Anything Model for precise table region detection
- **LLM Classification**: Text-based page and table type classification with Qwen3-8B
- **Dual Input**: Image for SAM3 detection + PDF text for LLM classification
- **Agent Integration**: Serves as backend service for automatic page selection in Agent UI
- **SageMaker Ready**: Containerized for AWS SageMaker deployment
- **GPU Optimized**: CUDA 12.6 support, ml.g6.xlarge or better

## Architecture

```
Input (image_base64 + pdf_text)
         ↓
    ┌────┴────┐
    ↓         ↓
  SAM3      Ollama
  (detect)  (classify)
    ↓         ↓
    └────┬────┘
         ↓
   Output (bboxes + types)
```

- **SAM3**: Table region detection from image → bounding boxes
- **Ollama (Qwen3-8B)**: Text classification → page_type, table_type
- **FastAPI**: REST API server

## Quick Start - Local

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Models

```bash
bash scripts/download_sam3_model.sh      # → models/sam3/checkpoints/
bash scripts/download_ollama_models.sh   # → models/ollama/models/
```

### 3. Run Locally with Docker

```bash
bash scripts/test-docker-local.sh
```

This starts the service on `http://127.0.0.1:8080`

### 4. Test the API

```bash
python3 scripts/test_local_api.py test_image.png [test_text.txt]
```

## API Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "sam3_ready": true,
  "ollama_ready": true
}
```

### POST /api/v1/analyze-page

Analyze a page with table detection and classification.

**Request:**
```json
{
  "image_base64": "base64-encoded-image-data",
  "pdf_text": "extracted text from PDF document"
}
```

**Response:**
```json
{
  "page_type": "balance_sheet",
  "confidence_page_type": 0.95,
  "table_type": "financial_table",
  "confidence_table_type": 0.87,
  "bboxes": [
    {
      "x1": 100,
      "y1": 150,
      "x2": 500,
      "y2": 400,
      "confidence": 0.92
    }
  ],
  "image_width": 1000,
  "image_height": 1500,
  "metadata": {
    "processing_time": 2.34,
    "sam3_tables_detected": 1,
    "pdf_text_length": 1234
  }
}
```

### POST /api/v1/analyze-page-file

Analyze a page from uploaded file.

**Request:**
- Form data with `file` (image) and `pdf_text` (query parameter)

**Response:** Same as `/api/v1/analyze-page`

## SageMaker Deployment

### Prerequisites

- AWS Account with SageMaker access
- ECR repository created
- S3 bucket for model storage
- IAM role for SageMaker execution

### Deployment Steps

1. **Build and push Docker image to ECR:**
   ```bash
   bash scripts/build_and_push_ecr.sh
   ```

2. **Download models locally:**
   ```bash
   bash scripts/download_sam3_model.sh
   bash scripts/download_ollama_models.sh
   ```

3. **Pack models to S3:**
   ```bash
   bash scripts/pack_model_to_s3.sh
   ```
   Models stored at: `s3://table-analysis-storage-models/model.tar.gz`

4. **Deploy endpoint:**
   ```bash
   bash scripts/deploy.sh
   ```
   Interactive prompts for AWS configuration

5. **Test deployed endpoint:**
   ```bash
   python3 scripts/test_online_api.py https://<sagemaker-endpoint> image.png
   ```

## Configuration

### config.yaml

Located at `/opt/program/config.yaml` in container.

**Key Settings:**
```yaml
models:
  base_path: /opt/ml/model
  sam3:
    checkpoint_dir: /opt/ml/model/sam3/checkpoints
    device: cuda
    confidence_threshold: 0.5
    text_prompt: financial table
  ollama:
    host: http://127.0.0.1:11434
    model: qwen3:8b
    timeout: 60
```

### Environment Variables (SageMaker)

- `SM_MODEL_DIR` - Model directory (default: `/opt/ml/model`)
- `OLLAMA_HOST` - Ollama API endpoint (default: `http://127.0.0.1:11434`)
- `OLLAMA_MODELS` - Ollama models path (default: `/opt/ml/model/ollama/models`)
- `SAM3_CHECKPOINT_DIR` - SAM3 checkpoints path (default: `/opt/ml/model/sam3/checkpoints`)

## Testing

### Local Testing

```bash
# Test with local Docker container
python3 scripts/test_local_api.py <image_path> [<text_file>]

# Examples
python3 scripts/test_local_api.py test.png                    # uses default PDF text
python3 scripts/test_local_api.py test.png document.txt       # uses custom text
```

### Online Testing (SageMaker)

```bash
# Test deployed endpoint
python3 scripts/test_online_api.py <endpoint_url> <image_path> [<text_file>]

# Examples
python3 scripts/test_online_api.py http://localhost:8080 test.png
python3 scripts/test_online_api.py https://sagemaker-endpoint.com image.png document.txt
```

## Storage Locations

### Local Development
- SAM3 models: `models/sam3/checkpoints/`
- Ollama models: `models/ollama/models/`

### SageMaker Container
- Base path: `/opt/ml/model`
- SAM3: `/opt/ml/model/sam3/checkpoints/`
- Ollama: `/opt/ml/model/ollama/models/`

### S3 (Production)
- Model archive: `s3://table-analysis-storage-models/model.tar.gz`
- Docker image: ECR repository `financial-table-analysis`

## Requirements (Local)

- Python 3.10+
- PyTorch 2.7.0+ with CUDA 12.6
- Ollama
- Docker 
- 16GB+ GPU VRAM

## Project Structure

```
services/
├── src/
│   ├── main.py              # FastAPI application
│   ├── analyzer.py          # SAM3 + Ollama orchestration
│   ├── sam3_detector.py     # SAM3 table detection
│   ├── ollama_client.py     # Ollama LLM client
│   ├── pdf_processor.py     # PDF text extraction
│   ├── models.py            # Pydantic schemas
│   ├── config.py            # Configuration loading
│   └── __init__.py
├── scripts/
│   ├── download_sam3_model.sh       # Download SAM3 weights
│   ├── download_ollama_models.sh    # Download Ollama models
│   ├── pack_model_to_s3.sh          # Package models to S3
│   ├── build_and_push_ecr.sh        # Build and push Docker image
│   ├── deploy.sh                    # Deploy to SageMaker
│   ├── deploy_endpoint.py           # SageMaker deployment script
│   ├── test-docker-local.sh         # Run local Docker test
│   ├── test_local_api.py            # Local API test client
│   └── test_online_api.py           # Remote API test client
├── Dockerfile                       # Container image definition
├── serve                            # Container entrypoint
├── config.yaml                      # Service configuration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```
