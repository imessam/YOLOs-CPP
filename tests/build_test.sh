#!/bin/bash
# ============================================================================
# YOLOs-CPP Test Build Script (Wrapper)
# ============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Default values
TEST_TASK="${1:-0}"           # 0=detection, 1=classification, 2=segmentation, 3=pose, 4=obb, 5=all
ONNXRUNTIME_VERSION="${2:-1.20.1}"
ONNXRUNTIME_GPU="${3:-0}"     # 0=CPU, 1=GPU

echo "============================================"
echo "  YOLOs-CPP Test Build (Delegating to Root)"
echo "============================================"
echo "  Task:          $TEST_TASK"
echo "  ONNX Runtime:  $ONNXRUNTIME_VERSION"
echo "  GPU:           $ONNXRUNTIME_GPU"
echo "============================================"

# Invoke root build script with test environment
# We assume root build script is in scripts/build.sh
bash "${PROJECT_ROOT}/scripts/build.sh" "$ONNXRUNTIME_VERSION" "$ONNXRUNTIME_GPU"

echo ""
echo "Build complete! Test executables are in build/tests/"
