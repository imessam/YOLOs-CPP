cmake_minimum_required(VERSION 3.0.0)
project(yolo_ort)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime
set(ONNXRUNTIME_VERSION "21")
set(ONNXRUNTIME_DIR "/home/essam/dev/libs/onnx/onnxruntime-linux-x64-gpu-1.21.0")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
include_directories("${ONNXRUNTIME_DIR}/include")
add_compile_definitions("ONNXRUNTIME_VERSION=${ONNXRUNTIME_VERSION}")

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA 10.2 REQUIRED)

# set(CMAKE_CUDA_STANDARD 10.1)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# !CUDA

find_package(OpenCV REQUIRED)

set(DEPS 
  ${OpenCV_LIBS}
)

set(EXECS)
set(LIBS)
set(TESTS)

## Libraries setup
include(${CMAKE_CURRENT_LIST_DIR}/src/CMakeLists.txt)

include(${CMAKE_CURRENT_LIST_DIR}/libs/CMakeLists.txt)

set(LIBS ObjectDetectionONNXLib ${libs_targets_object_detection_onnx})

foreach(_lib
${LIBS})
  target_link_libraries(${_lib}
      ${DEPS}
    )
endforeach()


## Executables setup
# Add executable for image inference
add_executable(image_inference
               src/image_inference.cpp)
set(EXECS ${EXECS} image_inference)


# Add executable for camera inference
add_executable(camera_inference
               src/camera_inference.cpp)
set(EXECS ${EXECS} camera_inference)

# Add executable for video inference
add_executable(video_inference
               src/video_inference.cpp)
set(EXECS ${EXECS} video_inference)


foreach(_target
${EXECS})
  target_link_libraries(${_target}
    ${DEPS}
    ${LIBS}
    )

  target_include_directories(${_target} PRIVATE "${ONNXRUNTIME_DIR}/include")
  target_compile_features(${_target} PRIVATE cxx_std_14)

  if(UNIX)
    message(STATUS "We are building on Linux!")
    target_link_libraries(${_target} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
  endif(UNIX)

  if(APPLE)
    message(STATUS "We are building on macOS!")
    target_link_libraries(${_target} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
  endif(APPLE)

  if(WIN32)
    message(STATUS "We are building on Windows!")
    target_link_libraries(${_target} "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
  endif(WIN32)


endforeach()