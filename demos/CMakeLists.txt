# ============================================================================
# YOLOs-CPP Demo Applications
# ============================================================================

# Helper Function to add demo executables
# This function is defined in the root CMakeLists.txt but we call it here
# Note: we need to adjust paths as we are now in the demos/ subdirectory

# Main Executables
add_yolo_executable(image_inference image_inference.cpp)
add_yolo_executable(video_inference video_inference.cpp)
add_yolo_executable(camera_inference camera_inference.cpp)
add_yolo_executable(class_image_inference class_image_inference.cpp)
add_yolo_executable(batch_image_inference batch_image_inference.cpp)
