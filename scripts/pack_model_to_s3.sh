#!/bin/bash
set -e

AWS_REGION=${AWS_REGION:-ap-northeast-1}

# Generate unique bucket name with timestamp (not saved to config)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BUCKET_NAME="table-analysis-models-${TIMESTAMP}"
S3_URI="s3://${BUCKET_NAME}/model.tar.gz"

echo "Packing models to S3"
echo "Region: $AWS_REGION"
echo "Generated Bucket: $BUCKET_NAME"
echo "Target: $S3_URI"
echo

[ -d "models/sam3/checkpoints" ] || { echo "ERROR: SAM3 not found"; exit 1; }
[ -d "models/ollama/models" ] || { echo "ERROR: Ollama not found"; exit 1; }

tar -C models -czf models/model.tar.gz sam3/checkpoints ollama/models
du -sh models/model.tar.gz

# Create bucket if not exists
aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null || \
aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$AWS_REGION" \
  --create-bucket-configuration LocationConstraint="$AWS_REGION"

aws s3 cp models/model.tar.gz "$S3_URI" --region "$AWS_REGION"
rm models/model.tar.gz

echo ""
echo "======================================"
echo "Upload complete!"
echo "======================================"
echo "S3 Model URI: ${S3_URI}"
echo "Bucket Name: $BUCKET_NAME"
echo "Region: $AWS_REGION"
echo ""

# Save S3 Model URI to file (for deploy.sh to read)
echo "${S3_URI}" > .s3_model_uri
echo "Saved to .s3_model_uri"
echo ""
echo "运行部署:"
echo "  bash ./deploy.sh"
echo "======================================"
