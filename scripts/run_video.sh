#!/bin/bash
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJECT_ROOT}/build/demos" && ./video_inference "$@"
