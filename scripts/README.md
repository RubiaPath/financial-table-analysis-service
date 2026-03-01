# Deployment Scripts

## Scripts

- **download_sam3_model.sh** - Download SAM3 from Hugging Face
- **download_ollama_models.sh** - Setup Ollama with Qwen3-8B
- **pack_model_to_s3.sh** - Package and upload models to S3
- **build_and_push_ecr.sh** - Build Docker image and push to ECR
- **deploy_endpoint.py** - Deploy to SageMaker endpoint
- **deploy.sh** - Interactive deployment wizard

## Quick Start

```bash
cd scripts/

bash download_sam3_model.sh
bash download_ollama_models.sh
bash pack_model_to_s3.sh my-bucket
bash deploy.sh
```

## Environment Variables

### Model Download

- `HF_TOKEN` - Hugging Face token for SAM3

### AWS Deployment

- `AWS_REGION` - AWS region (default: ap-northeast-1)
- `AWS_ACCOUNT_ID` - AWS account ID
- `ROLE_ARN` - SageMaker execution role ARN
- `IMAGE_URI` - ECR image URI
- `MODEL_DATA_URL` - S3 model path

See [../docs/](../docs/) for detailed documentation.
