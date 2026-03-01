#!/bin/bash
set -e

ENDPOINT_NAME=${ENDPOINT_NAME:-financial-table-analysis}
AWS_REGION=${AWS_REGION:-ap-northeast-1}
IMAGE_FILE=${1:-}

if [ -z "$IMAGE_FILE" ]; then
    echo "Usage: $0 <image_file>"
    echo "Example: $0 sample.png"
    exit 1
fi

if [ ! -f "$IMAGE_FILE" ]; then
    echo "ERROR: File not found: $IMAGE_FILE"
    exit 1
fi

IMAGE_BASE64=$(base64 -w0 "$IMAGE_FILE")

echo "Invoking endpoint: $ENDPOINT_NAME"
echo "Region: $AWS_REGION"
echo

aws sagemaker-runtime invoke-endpoint \
    --endpoint-name "$ENDPOINT_NAME" \
    --region "$AWS_REGION" \
    --content-type application/json \
    --body "{\"image_base64\": \"$IMAGE_BASE64\"}" \
    output.json

echo "Response saved to output.json"
cat output.json | python3 -m json.tool
