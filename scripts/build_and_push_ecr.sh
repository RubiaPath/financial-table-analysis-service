#!/usr/bin/env bash
# Build and push Docker image to ECR

set -euo pipefail

: "${AWS_REGION:=ap-northeast-1}"
: "${REPO_NAME:=financial-table-analysis}"
: "${TAG:=latest}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_URI="${ECR_REGISTRY}/${REPO_NAME}:${TAG}"

echo "Building and pushing to ECR..."
echo "ECR_URI: ${ECR_URI}"
echo ""

aws ecr describe-repositories \
    --repository-names "${REPO_NAME}" \
    --region "${AWS_REGION}" \
    >/dev/null 2>&1 || \
aws ecr create-repository \
    --repository-name "${REPO_NAME}" \
    --region "${AWS_REGION}"

aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"

docker build \
    --tag "${REPO_NAME}:${TAG}" \
    --build-arg SAM3_COMMIT=f6e51f59500a87c576c2df2323ce56b9fd7a12de \
    .

docker tag "${REPO_NAME}:${TAG}" "${ECR_URI}"
docker push "${ECR_URI}"

echo "Done: ${ECR_URI}"
