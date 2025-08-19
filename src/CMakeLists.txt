cmake_minimum_required(VERSION 3.15...3.31)

project(ObjectDetectionONNX VERSION 1.0
                  DESCRIPTION "Object Detection ONNX Module"
                  LANGUAGES CXX)

# add_compile_definitions("SOURCE_ROOT=${CMAKE_CURRENT_LIST_DIR}/../../")

## Libraries setup

set(OBJECT_DETECTION_ONNX_SOURCES

  ${CMAKE_CURRENT_LIST_DIR}/YOLO11.cpp
)

if (NOT TARGET ObjectDetectionONNXLib)

  include(${CMAKE_CURRENT_LIST_DIR}/../libs/CMakeLists.txt)

  add_library(ObjectDetectionONNXLib STATIC ${OBJECT_DETECTION_ONNX_SOURCES})

  target_include_directories(ObjectDetectionONNXLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include/det)
  target_include_directories(ObjectDetectionONNXLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include/tools)
  target_include_directories(ObjectDetectionONNXLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../libs/DetectionData/cpp/include)

  foreach(_lib_include_dir
  ${libs_include_object_detection_onnx})
  target_include_directories(ObjectDetectionONNXLib PUBLIC
        ${_lib_include_dir}
      )
  endforeach()

endif()
