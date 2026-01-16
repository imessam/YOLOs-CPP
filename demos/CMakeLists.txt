# CMake for demo applications

set(DEMO_SOURCES
    image_inference.cpp
    camera_inference.cpp
    video_inference.cpp
    class_image_inference.cpp
    batch_image_inference.cpp
)

foreach(demo_source ${DEMO_SOURCES})
    get_filename_component(demo_name ${demo_source} NAME_WE)
    add_executable(${demo_name} ${demo_source})
    target_link_libraries(${demo_name} PRIVATE yolo_ort_lib ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})
endforeach()
