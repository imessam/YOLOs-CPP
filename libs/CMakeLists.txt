cmake_minimum_required(VERSION 3.15...3.31)

project(ObjectDetectionONNX VERSION 1.0
                  DESCRIPTION "Object Detection ONNX Module"
                  LANGUAGES CXX)

## Libraries setup

set(libs_include_object_detection_onnx
  ${CMAKE_CURRENT_LIST_DIR}/../libs/DetectionData/cpp/include
)