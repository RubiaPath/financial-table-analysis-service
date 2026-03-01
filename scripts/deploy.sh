#!/bin/bash
set -e

echo "SageMaker Deployment"
echo

AWS_REGION="${AWS_REGION:-ap-northeast-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.g4dn.xlarge}"

echo -n "AWS Region [$AWS_REGION]: "
read -r input
AWS_REGION="${input:-$AWS_REGION}"

echo -n "AWS Account ID: "
read -r AWS_ACCOUNT_ID

echo -n "IAM Role ARN: "
read -r ROLE_ARN

echo -n "ECR Repository [financial-table-analysis]: "
read -r input
ECR_REPO="${input:-financial-table-analysis}"

echo -n "Image Tag [latest]: "
read -r input
IMAGE_TAG="${input:-latest}"

echo -n "S3 Bucket: "
read -r S3_BUCKET

echo -n "Instance Type [$INSTANCE_TYPE]: "
read -r input
INSTANCE_TYPE="${input:-$INSTANCE_TYPE}"

IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"
MODEL_DATA_URL="s3://$S3_BUCKET/models/model.tar.gz"

echo
echo "Configuration:"
echo "  Region: $AWS_REGION"
echo "  Account: $AWS_ACCOUNT_ID"
echo "  Image: $IMAGE_URI"
echo "  Models: $MODEL_DATA_URL"
echo
echo -n "Proceed? (y/N): "
read -r confirm

if [ "$confirm" != "y" ]; then
    exit 0
fi

echo && echo "Building Docker image..."
docker build -t "$ECR_REPO:$IMAGE_TAG" ..

echo && echo "Logging into ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

echo && echo "Pushing image..."
docker tag "$ECR_REPO:$IMAGE_TAG" "$IMAGE_URI"
docker push "$IMAGE_URI"

echo && echo "Downloading SAM3..."
bash ./download_sam3_model.sh

echo && echo "Downloading Ollama..."
bash ./download_ollama_models.sh

echo && echo "Packaging models..."
bash ./pack_model_to_s3.sh "$S3_BUCKET"

echo && echo "Deploying endpoint..."
export ROLE_ARN IMAGE_URI MODEL_DATA_URL AWS_REGION INSTANCE_TYPE
python3 deploy_endpoint.py

echo
echo "Done!"
