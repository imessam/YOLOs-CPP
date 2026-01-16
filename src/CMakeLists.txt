# Static library for YOLO implementations

# Add static library target with all YOLO implementation files
add_library(yolo_ort_lib STATIC
    YOLO.cpp
    YOLOCLASS.cpp
    YOLO-Seg.cpp
    YOLO-POSE.cpp
    YOLO-OBB.cpp
    YOLO-Utils.cpp
)

# Set include directories for the library
target_include_directories(yolo_ort_lib PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/../include"
    "${ONNXRUNTIME_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)