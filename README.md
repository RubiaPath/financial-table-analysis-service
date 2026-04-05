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

## Quick Start - Local Docker (Recommended)

### Minimum Steps to Success

```bash
# 1. Download models to project directory
bash scripts/download_sam3_model.sh      # → models/sam3/checkpoints/
bash scripts/download_ollama_models.sh   # → models/ollama/models/

# 2. Start service in Docker
bash scripts/test-docker-local.sh

# 3. Health check (in another terminal)
curl http://127.0.0.1:8080/health

# 4. Test single page analysis
python3 scripts/test_local_api.py test_image.png test_text.txt

# 5. Test batch PDF analysis
bash scripts/test_pdf_api.sh "./path/to/document.pdf"
```

The service runs entirely in Docker. No local Python installation required.

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
  "page_type": "main",
  "confidence_page_type": 0.95,
  "table_type": "BALANCE_SHEET",
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

**Valid page_type values:** `main`, `supplement`, `other`
**Valid table_type values:** `BALANCE_SHEET`, `INCOME_STATEMENT`, `CASH_FLOW`, `EQUITY`, `NOTES`, `OTHER_FINANCIAL`, `NON_FINANCIAL`, `UNKNOWN`

### POST /api/v1/analyze-page-file

Analyze a page from uploaded file.

**Request:**
- Form data with `file` (image) and `pdf_text` (query parameter)

**Response:** Same as `/api/v1/analyze-page`

### POST /api/v1/analyze-pdf

Batch analyze all pages in a PDF document with complete pipeline (SAM3 table detection + per-table Ollama classification).

**Request:**
- Form data with `file` (PDF file)

**Response:**
```json
{
  "total_pages": 100,
  "pages_with_tables": 42,
  "pages": [
    {
      "page_number": 1,
      "page_type": "other",
      "confidence_page_type": 0.0,
      "tables": [],
      "image_height": 1650,
      "image_width": 1275,
      "pdf_text": "Annual Report 2023..."
    },
    {
      "page_number": 7,
      "page_type": "main",
      "confidence_page_type": 0.92,
      "tables": [
        {
          "x1": 50,
          "y1": 100,
          "x2": 500,
          "y2": 600,
          "confidence": 0.89,
          "table_type": "BALANCE_SHEET",
          "confidence_table_type": 0.87
        },
        {
          "x1": 50,
          "y1": 650,
          "x2": 500,
          "y2": 1200,
          "confidence": 0.91,
          "table_type": "BALANCE_SHEET",
          "confidence_table_type": 0.84
        }
      ],
      "image_height": 1650,
      "image_width": 1275,
      "pdf_text": "ASSETS Current assets..."
    }
  ],
  "metadata": {
    "num_pages": 100,
    "pages_with_tables": 42,
    "sam3_prompt": "financial table",
    "ollama_model": "qwen3:8b",
    "classification_method": "Per-table individual classification"
  }
}
```

**Processing Pipeline:**
1. For each page: extract text and render to image (1.5x upscaling)
2. **SAM3 detection**: Find table regions → bboxes
3. **If tables found**:
   - Classify page type using Ollama (LLM on extracted text)
   - For each detected table: classify table type individually with Ollama
   - Return page_type + per-table classifications
4. **If NO tables found**:
   - Skip Ollama calls (optimization)
   - Return `page_type: "other"` with `confidence: 0.0`
   - Return empty tables list

**Intent:** Pages without tables are marked as "other" to indicate "no financial content detected" rather than "unknown classification"

**Optimization:** Ollama LLM calls are expensive; only pages with detected tables trigger classification

## SageMaker Deployment

### Prerequisites

- AWS Account with SageMaker access
- ECR repository created
- IAM role for SageMaker execution
- AWS CLI configured with appropriate credentials

### Standard Deployment (Recommended)

Three independent steps:

#### 1. Build and push Docker image to ECR (one-time)

```bash
bash scripts/build_and_push_ecr.sh
```

image is now in ECR and won't change unless you rebuild.

#### 2. Package models to S3

```bash
bash scripts/pack_model_to_s3.sh
```

- Generates unique bucket: `table-analysis-models-YYYYMMDD-HHMMSS`
- Uploads tar.gz of SAM3 + Ollama models (~15GB)
- Saves S3 Model URI to `.s3_model_uri` (gitignored)
- Output: `S3 Model URI: s3://table-analysis-models-20260406-151000/model.tar.gz`

**Note:** Models are versioned by timestamp. Rerun this if models change.

**Optional bucket naming:** For multi-environment/account setups, you can customize the bucket name in the script to include environment or account ID for better organization.

#### 3. Deploy SageMaker endpoint

```bash
bash scripts/deploy.sh
```

- Reads S3 Model URI from `.s3_model_uri` (created in step 2)
- Creates SageMaker: Model → EndpointConfig → Endpoint
- Uses existing ECR image (from step 1)
- Does NOT rebuild the Docker image

**Note:** If only models changed (step 2), rerun step 3. If code/dependencies changed, rerun both steps 1 and 3.

#### 4. Test endpoint

```bash
python3 scripts/test_online_api.py https://<sagemaker-endpoint-url> image.png
```

To get the endpoint URL after deployment, check AWS SageMaker Console → Endpoints.

### Full Workflow (One-time setup)

```bash
# Download models
bash scripts/download_sam3_model.sh
bash scripts/download_ollama_models.sh

# Build Docker image and push to ECR
bash scripts/build_and_push_ecr.sh

# Package models to S3
bash scripts/pack_model_to_s3.sh

# Deploy SageMaker endpoint
bash scripts/deploy.sh

# Test
python3 scripts/test_online_api.py https://<endpoint> test.png
```

### Updating Deployed Service

**If only ML models changed:**
```bash
bash scripts/pack_model_to_s3.sh
bash scripts/deploy.sh
```

**If code/dependencies changed:**
```bash
bash scripts/build_and_push_ecr.sh
bash scripts/pack_model_to_s3.sh
bash scripts/deploy.sh
```

## Configuration

### Docker Runtime

The service runs entirely in Docker:
- **Ollama**: Included in container, starts automatically in background
- **SAM3**: Loads from `/opt/ml/model/sam3/checkpoints/` at startup
- **FastAPI**: Listens on port 8080

When using `test-docker-local.sh`:
- Ollama starts inside the container
- Both Ollama and FastAPI are ready once health check returns success

### config.yaml

Located at `/opt/program/config.yaml` in container (for reference).

**Key paths:**
```yaml
models:
  base_path: /opt/ml/model           # SageMaker standard location
  sam3:
    checkpoint_dir: /opt/ml/model/sam3/checkpoints    # HF cache format
  ollama:
    host: http://127.0.0.1:11434     # Internal container network
```

### Environment Variables (SageMaker)

- `SM_MODEL_DIR` - Model directory (default: `/opt/ml/model`)
- `OLLAMA_HOST` - Ollama API endpoint (default: `http://127.0.0.1:11434`)
- `OLLAMA_MODELS` - Ollama models path (default: `/opt/ml/model/ollama/models`)
- `SAM3_CHECKPOINT_DIR` - SAM3 checkpoints path (default: `/opt/ml/model/sam3/checkpoints`)

## Testing

### Local Testing

**Single page analysis:**
```bash
python3 scripts/test_local_api.py <image_path> [<text_file>]

# Examples
python3 scripts/test_local_api.py test.png                    # uses default PDF text
python3 scripts/test_local_api.py test.png document.txt       # uses custom text
```

**Batch PDF analysis (curl + bash):**
```bash
bash scripts/test_pdf_api.sh <pdf_file>

# Examples
bash scripts/test_pdf_api.sh "./pdf/3M 2022 Annual Report_Updated.pdf"
bash scripts/test_pdf_api.sh "./documents/annual_report.pdf"
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
- SAM3 models: `models/sam3/checkpoints/` (HuggingFace cache format: `blobs/` + `snapshots/<hash>/`)
- Ollama models: `models/ollama/models/`
- S3 Model URI cache: `.s3_model_uri` (auto-generated, added to .gitignore)

### SageMaker Container
- Base path: `/opt/ml/model`
- SAM3: `/opt/ml/model/sam3/checkpoints/` (supports HF cache format)
- Ollama: `/opt/ml/model/ollama/models/`

### S3 (Production)
- Model archive: `s3://table-analysis-models-{TIMESTAMP}/model.tar.gz` (auto-generated unique bucket)
- Docker image: ECR repository `financial-table-analysis`

### HuggingFace Cache Format
SAM3 checkpoint supports standard HuggingFace cache structure:
- `checkpoints/blobs/` - Model files
- `checkpoints/snapshots/<hash>/` - Snapshot pointers
- Supports both gated repos and offline mode

## Requirements (Local)

- Python 3.10+
- PyTorch 2.7.0+ with CUDA 12.6
- Ollama
- Docker 
- 16GB+ GPU VRAM

## Project Structure

```
.
├── src/
│   ├── main.py              # FastAPI application  
│   ├── analyzer.py          # SAM3 + Ollama orchestration
│   ├── sam3_detector.py     # SAM3 table detection wrapper
│   ├── ollama_client.py     # Ollama LLM client
│   ├── models.py            # Pydantic schemas (request/response)
│   ├── config.py            # Configuration loading
│   └── __init__.py
├── scripts/
│   ├── download_sam3_model.sh       # Download SAM3 weights to models/
│   ├── download_ollama_models.sh    # Download Ollama models to models/
│   ├── pack_model_to_s3.sh          # Package to S3, save URI to .s3_model_uri
│   ├── build_and_push_ecr.sh        # Build Docker image, push to ECR
│   ├── deploy.sh                    # Deploy SageMaker endpoint (uses ECR image + S3 models)
│   ├── deploy_endpoint.py           # SageMaker boto3 automation
│   ├── test-docker-local.sh         # Local Docker: start Ollama + FastAPI
│   ├── test_local_api.py            # Test single page (/api/v1/analyze-page)
│   ├── test_pdf_api.sh              # Test batch PDF (/api/v1/analyze-pdf)  
│   └── test_online_api.py           # Test SageMaker endpoint
├── models/
│   ├── sam3/checkpoints/            # SAM3 (HF cache format, created by download script)
│   └── ollama/models/               # Ollama (created by download script)
├── Dockerfile                       # Container image definition
├── serve                            # Container entrypoint (fastapi + ollama)
├── config.yaml                      # Service configuration (paths, prompts, categories)
├── requirements.txt                 # Python dependencies
├── .gitignore
├── README.md                        # This file
└── .s3_model_uri                    # Auto-generated S3 model URI (gitignored)
```
