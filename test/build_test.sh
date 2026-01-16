#!/bin/bash

set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)

# Default values
TEST_TASK="${1:-0}" # 0: detection, 1: classification, 2: segmentation, 3: pose, 4: obb, 5: all
ONNXRUNTIME_VERSION="${2:-1.20.1}"
ONNXRUNTIME_GPU="${3:-0}"


# Function to display usage
usage() {
    echo "Usage: $0 [TEST_TASK] [ONNXRUNTIME_VERSION] [ONNXRUNTIME_GPU]"
    echo
    echo "This script downloads ONNX Runtime for the current platform and architecture and builds YOLOs-CPP."
    echo
    echo "Arguments:"
    echo "  TEST_TASK            The test task to build (0 for detection, 1 for classification, 2 for segmentation, 3 for pose, 4 for obb, 5 for all, default: 0)."
    echo "  ONNXRUNTIME_VERSION   Version of ONNX Runtime to download (default: 1.20.1)."
    echo "  ONNXRUNTIME_GPU       Whether to use GPU support (0 for CPU, 1 for GPU, default: 0)."
    echo
    echo "Examples:"
    echo "  $0 0 1.20.1 0          # Downloads ONNX Runtime v1.20.1 for CPU and builds detection tests."
    echo "  $0 5 1.16.3 1        # Downloads ONNX Runtime v1.16.3 for GPU and builds all tests."
    echo
    exit 1
}

# Show usage if help is requested
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

PROJECT_ROOT=$(cd "${CURRENT_DIR}/.." && pwd)

# Function to setup ONNX Runtime using the centralized script
setup_dependencies() {
    echo "Ensuring ONNX Runtime is set up..."
    if [ -f "${PROJECT_ROOT}/scripts/setup_onnxruntime.sh" ]; then
        bash "${PROJECT_ROOT}/scripts/setup_onnxruntime.sh" "${ONNXRUNTIME_VERSION}" "${ONNXRUNTIME_GPU}"
    else
        echo "Error: scripts/setup_onnxruntime.sh not found."
        exit 1
    fi
}

# Determine the expected ONNX Runtime directory name (must match setup_onnxruntime.sh)
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
    x86_64) ONNXRUNTIME_ARCH="x64" ;;
    *) echo "Unsupported architecture: $architecture"; exit 1 ;;
esac

ONNXRUNTIME_DIR_NAME="onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}"
if [[ "$ONNXRUNTIME_GPU" -eq 1 ]]; then
    ONNXRUNTIME_DIR_NAME="${ONNXRUNTIME_DIR_NAME}-gpu"
fi
ONNXRUNTIME_DIR_NAME="${ONNXRUNTIME_DIR_NAME}-${ONNXRUNTIME_VERSION}"
ONNXRUNTIME_DIR="${PROJECT_ROOT}/${ONNXRUNTIME_DIR_NAME}"

# Function to build the project
build_project() {
    local build_type="${1:-Release}"
    local build_dir="${CURRENT_DIR}/build"

    # Ensure the build directory exists
    mkdir -p "$build_dir"
    cd "$build_dir"

    echo "Configuring CMake with build type: $build_type ..."
    echo "Using ONNX Runtime from: ${ONNXRUNTIME_DIR}"
    
    cmake .. -D ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}" \
             -DtestTask="${TEST_TASK}" \
             -DCMAKE_BUILD_TYPE="$build_type" \
             -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native"

    echo "Building tests incrementally ..."
    cmake --build . -- -j$(nproc)
}

# Main script execution
setup_dependencies
build_project "Release"

echo "Test build completed successfully."