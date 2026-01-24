# ============================================================================
# YOLOs-CPP - High-Performance C++ Inference for YOLO Models
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(YOLOs-CPP VERSION 2.0.0 LANGUAGES CXX)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Dependencies (OpenCV, ONNX Runtime, CUDA, GTest)
# ============================================================================
include(cmake/common.cmake)

# Ensure ONNXRUNTIME_LIB variable matches what common.cmake provides for compatibility
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_LIBRARIES})

message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_DIR}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")


# ============================================================================
# Static Library
# ============================================================================
# Threading support
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_subdirectory(src)

# Link yolos library dependencies here
target_link_libraries(yolos PUBLIC 
    ${OpenCV_LIBS}
    Threads::Threads
)
target_link_libraries(yolos PRIVATE 
    ${ONNXRUNTIME_LIB}
)


# ============================================================================
# Helper Function
# ============================================================================
function(add_yolo_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE 
        yolos
        ${ONNXRUNTIME_LIB}
    )
    
    # Platform-specific runtime library handling
    if(WIN32)
        # Copy ONNX Runtime DLL to output directory
        set(ONNXRUNTIME_DLL "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
        if(EXISTS "${ONNXRUNTIME_DLL}")
            add_custom_command(TARGET ${name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_DLL}"
                "$<TARGET_FILE_DIR:${name}>"
            )
        endif()
    elseif(UNIX AND NOT APPLE)
        # Set RPATH for runtime library discovery
        set_target_properties(${name} PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
        )
    elseif(APPLE)
        # Set RPATH for macOS
        set_target_properties(${name} PROPERTIES
            BUILD_RPATH "@executable_path;${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "@executable_path;${ONNXRUNTIME_DIR}/lib"
        )
    endif()
endfunction()

# ============================================================================
# Demos
# ============================================================================
add_subdirectory(demos)


# ============================================================================
# Examples (optional)
# ============================================================================
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Tests (optional)
# ============================================================================
option(ENABLE_TESTS "Build test suite" OFF)
if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()


# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== YOLOs-CPP Configuration ===")
message(STATUS "  Version:       ${PROJECT_VERSION}")
message(STATUS "  Build Type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV:        ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime:  ${ONNXRUNTIME_DIR}")
message(STATUS "  Examples:      ${BUILD_EXAMPLES}")
message(STATUS "")
