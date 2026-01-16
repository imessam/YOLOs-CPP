cmake_minimum_required(VERSION 3.15...3.31)

project(Common VERSION 1.0
                  DESCRIPTION "Common deps"
                  LANGUAGES CXX)


# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA 10.2 REQUIRED)

# set(CMAKE_CUDA_STANDARD 10.1)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# !CUDA

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

# ONNX Runtime discovery
if(NOT ONNXRUNTIME_DIR)
    # Use CMAKE_CURRENT_LIST_DIR to find the project root regardless of CMAKE_SOURCE_DIR
    set(_project_root "${CMAKE_CURRENT_LIST_DIR}/..")
    
    # Try GPU version first as it's often preferred if available
    if(EXISTS "${_project_root}/onnxruntime-linux-x64-gpu-1.20.1")
        set(ONNXRUNTIME_DIR "${_project_root}/onnxruntime-linux-x64-gpu-1.20.1")
    elseif(EXISTS "${_project_root}/onnxruntime-linux-x64-1.20.1")
        set(ONNXRUNTIME_DIR "${_project_root}/onnxruntime-linux-x64-1.20.1")
    else()
        # Fallback to the GPU name as a placeholder if nothing exists
        set(ONNXRUNTIME_DIR "${_project_root}/onnxruntime-linux-x64-gpu-1.20.1")
    endif()
endif()
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

set(ONNXRUNTIME_VERSION "21")

include_directories("${ONNXRUNTIME_DIR}/include")
add_compile_definitions("ONNXRUNTIME_VERSION=${ONNXRUNTIME_VERSION}")
# !ONNX Runtime

# Platform-specific ONNX Runtime linking setup
if(UNIX AND NOT APPLE)
    set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
elseif(APPLE)
    set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
elseif(WIN32)
    set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_DIR}/lib/onnxruntime_gpu.lib") # Usually onnxruntime.lib or onnxruntime_gpu.lib
endif()
set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARIES} CACHE INTERNAL "ONNX Runtime libraries")

find_package(PkgConfig REQUIRED)



include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
