#!/usr/bin/env bash
# Download Ollama models

set -euo pipefail

: "${OLLAMA_MODEL:=qwen3:8b}"
: "${WORKDIR:=models/ollama/models}"

mkdir -p "${WORKDIR}"

if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama not installed"
    echo "Install: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

export OLLAMA_MODELS="${WORKDIR}"
echo "Pulling ${OLLAMA_MODEL}..."
ollama pull "${OLLAMA_MODEL}"

echo "Done: ${WORKDIR}"
