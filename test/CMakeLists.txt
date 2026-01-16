cmake_minimum_required(VERSION 3.15)

# Detect if built as a standalone project or as part of the main project
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(yolo_ort_test)
    set(STANDALONE_BUILD ON)
    
    # Include project-wide configuration and dependencies
    include("../cmake/common.cmake")
    
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Build the core library as part of this standalone build
    add_subdirectory("../src" "src_lib")
else()
    set(STANDALONE_BUILD OFF)
endif()

add_compile_definitions("BASE_PATH=${CMAKE_CURRENT_SOURCE_DIR}/")
add_compile_definitions("BASE_PATH_DETECTION=${CMAKE_CURRENT_SOURCE_DIR}/detection/")
add_compile_definitions("BASE_PATH_CLASSIFICATION=${CMAKE_CURRENT_SOURCE_DIR}/classification/")
add_compile_definitions("BASE_PATH_SEGMENTATION=${CMAKE_CURRENT_SOURCE_DIR}/segmentation/")
add_compile_definitions("BASE_PATH_POSE=${CMAKE_CURRENT_SOURCE_DIR}/pose/")
add_compile_definitions("BASE_PATH_OBB=${CMAKE_CURRENT_SOURCE_DIR}/obb/")

# nlohmann_json
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

# GoogleTest (only if standalone, otherwise main project handles it)
if(STANDALONE_BUILD)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

enable_testing()
include(GoogleTest)

option(testTask "Specify which test task to build: 0 for detection, 1 for classification, 2 for segmentation, 3 for pose, 4 for obb, 5 for all." STRING)
set(detection_path "${CMAKE_CURRENT_SOURCE_DIR}/detection")
set(classification_path "${CMAKE_CURRENT_SOURCE_DIR}/classification")
set(segmentation_path "${CMAKE_CURRENT_SOURCE_DIR}/segmentation")
set(pose_path "${CMAKE_CURRENT_SOURCE_DIR}/pose")
set(obb_path "${CMAKE_CURRENT_SOURCE_DIR}/obb")

set(EXES)
set(TESTS)

# Add executables
if (testTask STREQUAL "0" OR testTask STREQUAL "5")
    add_executable(inference_detection_cpp ${detection_path}/inference_detection_cpp.cpp)
    add_executable(compare_detection_results ${detection_path}/compare_results.cpp)
    list(APPEND EXES inference_detection_cpp)
    list(APPEND TESTS compare_detection_results)
endif()

if (testTask STREQUAL "1" OR testTask STREQUAL "5")
    add_executable(inference_classification_cpp ${classification_path}/inference_classification_cpp.cpp)
    add_executable(compare_classification_results ${classification_path}/compare_results.cpp)
    list(APPEND EXES inference_classification_cpp)
    list(APPEND TESTS compare_classification_results)
endif()

if (testTask STREQUAL "2" OR testTask STREQUAL "5")
    add_executable(inference_segmentation_cpp ${segmentation_path}/inference_segmentation_cpp.cpp)
    add_executable(compare_segmentation_results ${segmentation_path}/compare_results.cpp)
    list(APPEND EXES inference_segmentation_cpp)
    list(APPEND TESTS compare_segmentation_results)
endif()

if (testTask STREQUAL "3" OR testTask STREQUAL "5")
    add_executable(inference_pose_cpp ${pose_path}/inference_pose_cpp.cpp)
    add_executable(compare_pose_results ${pose_path}/compare_results.cpp)
    list(APPEND EXES inference_pose_cpp)
    list(APPEND TESTS compare_pose_results)
endif()

if (testTask STREQUAL "4" OR testTask STREQUAL "5")
    add_executable(inference_obb_cpp ${obb_path}/inference_obb_cpp.cpp)
    add_executable(compare_obb_results ${obb_path}/compare_obb_results.cpp)
    list(APPEND EXES inference_obb_cpp)
    list(APPEND TESTS compare_obb_results)
endif()

# Set dependencies for all executables
foreach(exec_target ${EXES})
    if(TARGET yolo_ort_lib)
        target_link_libraries(${exec_target} PRIVATE yolo_ort_lib ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES} nlohmann_json::nlohmann_json)
    else()
        # Fallback for standalone build if yolo_ort_lib isn't available
        target_link_libraries(${exec_target} PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES} nlohmann_json::nlohmann_json)
        target_include_directories(${exec_target} PRIVATE "${ONNXRUNTIME_DIR}/include")
    endif()
endforeach()

# Add tests linking
foreach(test_target ${TESTS})
    target_link_libraries(${test_target} PRIVATE
        nlohmann_json::nlohmann_json
        GTest::gtest_main
        ${OpenCV_LIBS}
    )
    gtest_discover_tests(${test_target})
endforeach()
