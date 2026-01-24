# ============================================================================
# YOLOs-CPP Static Library
# ============================================================================

# Source files
set(YOLO_SOURCES
    core/drawing.cpp
    core/nms.cpp
    core/preprocessing.cpp
    core/session_base.cpp
    core/types.cpp
    core/utils.cpp
    tasks/classification.cpp
    tasks/detection.cpp
    tasks/obb.cpp
    tasks/pose.cpp
    tasks/segmentation.cpp
)

# Create the library
add_library(yolos STATIC ${YOLO_SOURCES})

# Include directories
target_include_directories(yolos PUBLIC 
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
)

