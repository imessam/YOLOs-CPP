cmake_minimum_required(VERSION 3.15)
project(yolo_ort)

# Include common dependencies setup
include(cmake/common.cmake)

# Enforce C++17 (overriding common.cmake which sets 14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("include/")

# Add the src subdirectory to build the static library
add_subdirectory(src)

# Add the demos subdirectory for example applications
add_subdirectory(demos)

# Add the test subdirectory for unit and integration tests
add_subdirectory(test)