#!/usr/bin/env bash
# Pack models and upload to S3

set -euo pipefail

: "${AWS_REGION:=ap-northeast-1}"
: "${S3_URI:=s3://sam3-storage/models/model.tar.gz}"
: "${WORKDIR:=model_artifacts}"

echo "AWS_REGION: ${AWS_REGION}"
echo "S3_URI: ${S3_URI}"
echo ""

if [ ! -d "models/sam3/checkpoints" ]; then
    echo "ERROR: SAM3 checkpoints not found"
    exit 1
fi

if [ ! -d "models/ollama/models" ]; then
    echo "ERROR: Ollama models not found"
    exit 1
fi

rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"

cp -r models/sam3/checkpoints "${WORKDIR}/"
cp -r models/ollama/models "${WORKDIR}/"

cd "${WORKDIR}"
tar -czf model.tar.gz checkpoints/ models/
du -sh model.tar.gz
cd ..

aws s3 cp "${WORKDIR}/model.tar.gz" "${S3_URI}" --region "${AWS_REGION}"

echo "Upload complete: ${S3_URI}"
