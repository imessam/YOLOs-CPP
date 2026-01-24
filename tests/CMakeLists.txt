# ============================================================================
# YOLOs-CPP Test Suite
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(yolos_cpp_tests VERSION 2.0.0 LANGUAGES CXX)

# ============================================================================
# Dependencies (OpenCV, ONNX Runtime, nlohmann_json, GoogleTest)
# ============================================================================
include(../cmake/common.cmake)

# Match library variable used in parent
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_LIBRARIES})

# Support standalone build
if(NOT TARGET yolos)
    message(STATUS "YOLOs-CPP: Building tests in standalone mode")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_BINARY_DIR}/yolos_lib)
    
    # Standard link libraries for yolos if running standalone
    target_link_libraries(yolos PUBLIC 
        ${OpenCV_LIBS}
        Threads::Threads
    )
    target_link_libraries(yolos PRIVATE 
        ${ONNXRUNTIME_LIB}
    )
else()
    message(STATUS "YOLOs-CPP: Building tests in integrated mode")
endif()

# ============================================================================
# Base Path Definitions
# ============================================================================
add_compile_definitions("BASE_PATH=${CMAKE_CURRENT_SOURCE_DIR}/")
add_compile_definitions("BASE_PATH_DETECTION=${CMAKE_CURRENT_SOURCE_DIR}/detection/")
add_compile_definitions("BASE_PATH_CLASSIFICATION=${CMAKE_CURRENT_SOURCE_DIR}/classification/")
add_compile_definitions("BASE_PATH_SEGMENTATION=${CMAKE_CURRENT_SOURCE_DIR}/segmentation/")
add_compile_definitions("BASE_PATH_POSE=${CMAKE_CURRENT_SOURCE_DIR}/pose/")
add_compile_definitions("BASE_PATH_OBB=${CMAKE_CURRENT_SOURCE_DIR}/obb/")

message(STATUS "Test Base Path: ${CMAKE_CURRENT_SOURCE_DIR}")


# ============================================================================
# Test Task Selection
# ============================================================================
option(testTask "Specify test task: 0=detection, 1=classification, 2=segmentation, 3=pose, 4=obb, 5=all" STRING)
message(STATUS "Selected test task: ${testTask}")

set(detection_path "${CMAKE_CURRENT_SOURCE_DIR}/detection")
set(classification_path "${CMAKE_CURRENT_SOURCE_DIR}/classification")
set(segmentation_path "${CMAKE_CURRENT_SOURCE_DIR}/segmentation")
set(pose_path "${CMAKE_CURRENT_SOURCE_DIR}/pose")
set(obb_path "${CMAKE_CURRENT_SOURCE_DIR}/obb")

set(INFERENCE_EXES)
set(TEST_EXES)

# ============================================================================
# Detection Tests
# ============================================================================
if(testTask STREQUAL "0" OR testTask STREQUAL "5")
    add_executable(inference_detection_cpp 
        ${detection_path}/inference_detection_cpp.cpp)
    add_executable(compare_detection_results 
        ${detection_path}/compare_results.cpp)
    
    list(APPEND INFERENCE_EXES inference_detection_cpp)
    list(APPEND TEST_EXES compare_detection_results)
endif()

# ============================================================================
# Classification Tests
# ============================================================================
if(testTask STREQUAL "1" OR testTask STREQUAL "5")
    add_executable(inference_classification_cpp 
        ${classification_path}/inference_classification_cpp.cpp)
    add_executable(compare_classification_results 
        ${classification_path}/compare_results.cpp)
    
    list(APPEND INFERENCE_EXES inference_classification_cpp)
    list(APPEND TEST_EXES compare_classification_results)
endif()

# ============================================================================
# Segmentation Tests
# ============================================================================
if(testTask STREQUAL "2" OR testTask STREQUAL "5")
    add_executable(inference_segmentation_cpp 
        ${segmentation_path}/inference_segmentation_cpp.cpp)
    add_executable(compare_segmentation_results 
        ${segmentation_path}/compare_results.cpp)

    list(APPEND INFERENCE_EXES inference_segmentation_cpp)
    list(APPEND TEST_EXES compare_segmentation_results)
endif()

# ============================================================================
# Pose Tests
# ============================================================================
if(testTask STREQUAL "3" OR testTask STREQUAL "5")
    add_executable(inference_pose_cpp 
        ${pose_path}/inference_pose_cpp.cpp)
    add_executable(compare_pose_results 
        ${pose_path}/compare_results.cpp)
    
    list(APPEND INFERENCE_EXES inference_pose_cpp)
    list(APPEND TEST_EXES compare_pose_results)
endif()

# ============================================================================
# OBB Tests
# ============================================================================
if(testTask STREQUAL "4" OR testTask STREQUAL "5")
    add_executable(inference_obb_cpp 
        ${obb_path}/inference_obb_cpp.cpp)
    add_executable(compare_obb_results 
        ${obb_path}/compare_obb_results.cpp)
    
    list(APPEND INFERENCE_EXES inference_obb_cpp)
    list(APPEND TEST_EXES compare_obb_results)
endif()

# ============================================================================
# Configure Inference Executables
# ============================================================================
foreach(target ${INFERENCE_EXES})
    message(STATUS "Configuring inference target: ${target}")
    
    target_link_libraries(${target} PRIVATE
        yolos
        ${ONNXRUNTIME_LIB}
        nlohmann_json::nlohmann_json
    )
    
    # Set RPATH
    if(UNIX AND NOT APPLE)
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
        )
    endif()
endforeach()

# ============================================================================
# Configure Test Executables
# ============================================================================
foreach(target ${TEST_EXES})
    message(STATUS "Adding test target: ${target}")
    
    target_link_libraries(${target} PRIVATE
        nlohmann_json::nlohmann_json
        GTest::gtest_main
        ${OpenCV_LIBS}
    )
    
    gtest_discover_tests(${target})
endforeach()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== Test Suite Configuration ===")
message(STATUS "  Inference executables: ${INFERENCE_EXES}")
message(STATUS "  Test executables: ${TEST_EXES}")
message(STATUS "")
