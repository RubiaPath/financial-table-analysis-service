#!/usr/bin/env bash
# Download SAM3 model from Hugging Face

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN required}"
: "${REPO_ID:=facebook/sam3-large}"
: "${WORKDIR:=models/sam3/checkpoints}"

echo "Installing huggingface_hub..."
python3 -m pip install -q -U "huggingface_hub[cli]" 2>/dev/null || true

mkdir -p "${WORKDIR}"

echo "Downloading ${REPO_ID}..."
python3 << 'PYTHON'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["REPO_ID"],
    local_dir=os.environ["WORKDIR"],
    local_dir_use_symlinks=False,
    token=os.environ["HF_TOKEN"],
)
PYTHON

echo "Done: ${WORKDIR}"
