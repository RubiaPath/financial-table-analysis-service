#!/bin/bash
set -e

AWS_REGION=${AWS_REGION:-ap-northeast-1}
S3_URI=${S3_URI:-s3://table-analysis-storage-models/model.tar.gz}

echo "Packing models to S3"
echo "Region: $AWS_REGION"
echo "Target: $S3_URI"
echo

[ -d "models/sam3/checkpoints" ] || { echo "ERROR: SAM3 not found"; exit 1; }
[ -d "models/ollama/models" ] || { echo "ERROR: Ollama not found"; exit 1; }

tar -C models -czf model.tar.gz sam3/checkpoints ollama/models
du -sh model.tar.gz

aws s3 cp model.tar.gz "$S3_URI" --region "$AWS_REGION"
rm model.tar.gz

echo "Done"

echo "Upload complete: ${S3_URI}"
