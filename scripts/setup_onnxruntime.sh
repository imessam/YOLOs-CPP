#!/bin/bash

# setup_onnxruntime.sh - Downloads and extracts ONNX Runtime for the current platform

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Default values
ONNXRUNTIME_VERSION="${1:-1.20.1}"
ONNXRUNTIME_GPU="${2:-0}"

# Function to display usage
usage() {
    echo "Usage: $0 [ONNXRUNTIME_VERSION] [ONNXRUNTIME_GPU]"
    echo
    echo "This script downloads ONNX Runtime for the current platform and architecture."
    echo
    echo "Arguments:"
    echo "  ONNXRUNTIME_VERSION   Version of ONNX Runtime to download (default: 1.20.1)."
    echo "  ONNXRUNTIME_GPU       Whether to use GPU support (0 for CPU, 1 for GPU, default: 0)."
    echo
    echo "Examples:"
    echo "  $0 1.20.1 0          # Downloads ONNX Runtime v1.20.1 for CPU."
    echo "  $0 1.16.3 1          # Downloads ONNX Runtime v1.16.3 for GPU."
    echo
    exit 1
}

# Show usage if help is requested
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

# Detect platform and architecture
platform=$(uname -s)
architecture=$(uname -m)

case "$platform" in
Darwin*)
    ONNXRUNTIME_PLATFORM="osx"
    ONNXRUNTIME_ARCHIVE_EXTENSION="tgz"
    ;;
Linux*) 
    ONNXRUNTIME_PLATFORM="linux"
    ONNXRUNTIME_ARCHIVE_EXTENSION="tgz"
    ;;
MINGW*) 
    ONNXRUNTIME_PLATFORM="win"
    ONNXRUNTIME_ARCHIVE_EXTENSION="zip"
    ;;
*)
    echo "Unsupported platform: $platform"
    exit 1
    ;;
esac

# Determine ONNX Runtime architecture
case "$architecture" in
aarch64|arm64)
    ONNXRUNTIME_ARCH="aarch64"
    ;;
x86_64)
    ONNXRUNTIME_ARCH="x64"
    ;;
arm*)
    ONNXRUNTIME_ARCH="arm"
    ;;
i*86)
    ONNXRUNTIME_ARCH="x86"
    ;;
*)
    echo "Unsupported architecture: $architecture"
    exit 1
    ;;
esac

# Set the correct ONNX Runtime download filename
ONNXRUNTIME_FILE="onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}"
ONNXRUNTIME_DIR_NAME="onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}"

if [[ "$ONNXRUNTIME_GPU" -eq 1 ]]; then
    ONNXRUNTIME_FILE="${ONNXRUNTIME_FILE}-gpu"
    ONNXRUNTIME_DIR_NAME="${ONNXRUNTIME_DIR_NAME}-gpu"
fi

ONNXRUNTIME_FILE="${ONNXRUNTIME_FILE}-${ONNXRUNTIME_VERSION}.${ONNXRUNTIME_ARCHIVE_EXTENSION}"
ONNXRUNTIME_DIR_NAME="${ONNXRUNTIME_DIR_NAME}-${ONNXRUNTIME_VERSION}"
ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_FILE}"

# Absolute path for extraction
ONNXRUNTIME_FULL_PATH="${PROJECT_ROOT}/${ONNXRUNTIME_DIR_NAME}"

# Function to download and extract ONNX Runtime
download_onnxruntime() {
    if [ -d "$ONNXRUNTIME_FULL_PATH" ]; then
        echo "ONNX Runtime already exists at $ONNXRUNTIME_FULL_PATH. Skipping download."
        return 0
    fi

    echo "Downloading ONNX Runtime from $ONNXRUNTIME_URL ..."
    
    if ! curl -L -C - -o "${ONNXRUNTIME_FILE}" "$ONNXRUNTIME_URL"; then
        echo "Error: Failed to download ONNX Runtime."
        exit 1
    fi

    echo "Extracting ONNX Runtime ..."
    if [[ "${ONNXRUNTIME_ARCHIVE_EXTENSION}" = "tgz" ]]; then
        if ! tar -zxvf "${ONNXRUNTIME_FILE}" -C "$PROJECT_ROOT"; then
            echo "Error: Failed to extract ONNX Runtime."
            exit 1
        fi
    elif [[ "${ONNXRUNTIME_ARCHIVE_EXTENSION}" = "zip" ]]; then
        if ! unzip "${ONNXRUNTIME_FILE}" -d "$PROJECT_ROOT"; then
            echo "Error: Failed to extract ONNX Runtime."
            exit 1
        fi
    fi

    rm -f "${ONNXRUNTIME_FILE}"
    echo "ONNX Runtime setup complete in $ONNXRUNTIME_FULL_PATH"
}

download_onnxruntime
