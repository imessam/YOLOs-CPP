#!/bin/bash

# ============================================================================
# Build Script for YOLOs-CPP
# ============================================================================

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Default values
ONNXRUNTIME_VERSION="${1:-1.20.1}"
ONNXRUNTIME_GPU="${2:-0}"

# Function to display usage
usage() {
    echo "Usage: $0 [ONNXRUNTIME_VERSION] [ONNXRUNTIME_GPU]"
    echo
    echo "This script builds YOLOs-CPP. It will download ONNX Runtime if not found."
    echo
    echo "Arguments:"
    echo "  ONNXRUNTIME_VERSION   Version of ONNX Runtime to use (default: 1.20.1)."
    echo "  ONNXRUNTIME_GPU       Whether to use GPU support (0 for CPU, 1 for GPU, default: 0)."
    echo
    exit 1
}

# Show usage if help is requested
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

# Detect platform and architecture for directory name
platform=$(uname -s)
architecture=$(uname -m)

case "$platform" in
Darwin*) ONNXRUNTIME_PLATFORM="osx" ;;
Linux*)  ONNXRUNTIME_PLATFORM="linux" ;;
MINGW*)  ONNXRUNTIME_PLATFORM="win" ;;
*) echo "Unsupported platform: $platform"; exit 1 ;;
esac

case "$architecture" in
aarch64|arm64) ONNXRUNTIME_ARCH="aarch64" ;;
x86_64)        ONNXRUNTIME_ARCH="x64" ;;
arm*)          ONNXRUNTIME_ARCH="arm" ;;
i*86)          ONNXRUNTIME_ARCH="x86" ;;
*) echo "Unsupported architecture: $architecture"; exit 1 ;;
esac

# Construct the expected ONNX Runtime directory name
ONNXRUNTIME_DIR="${PROJECT_ROOT}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}"
if [[ "$ONNXRUNTIME_GPU" -eq 1 ]]; then
    ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}-gpu"
fi
ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}-${ONNXRUNTIME_VERSION}"

# Check for ONNX Runtime and download if missing
if [ ! -d "$ONNXRUNTIME_DIR" ]; then
    echo "ONNX Runtime not found at $ONNXRUNTIME_DIR"
    echo "Invoking download script..."
    bash "${PROJECT_ROOT}/scripts/download_onnxruntime.sh" "$ONNXRUNTIME_VERSION" "$ONNXRUNTIME_GPU"
fi

# Function to build the project
build_project() {
    local build_type="${1:-Release}"
    local build_dir="${PROJECT_ROOT}/build"

    # Ensure the build directory exists
    mkdir -p "$build_dir"
    cd "$build_dir"

    echo "Configuring CMake with build type: $build_type ..."
    cmake .. -D ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}" -DCMAKE_BUILD_TYPE="$build_type" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native"

    echo "Building project incrementally ..."
    cmake --build . -- -j$(nproc)
}

# Build the project
build_project "Release"

echo "Build completed successfully."
