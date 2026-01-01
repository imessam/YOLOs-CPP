#pragma once

// ===================================
// Single YOLOv11 Detector Header File
// ===================================
//
// This header defines the YOLODetector class for performing object detection using the YOLOv11 model.
// It includes necessary libraries, utility structures, and helper functions to facilitate model inference
// and result postprocessing.
//
// Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
// Date: 29.09.2024
//
// ================================

/**
 * @file YOLODetector.hpp
 * @brief Header file for the YOLODetector class, responsible for object detection
 *        using the YOLOv11 model with optimized performance for minimal latency.
 */

// Include necessary ONNX Runtime and OpenCV headers
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "detection.h"

// Include debug and custom ScopedTimer tools for performance measurement
#ifdef DEBUG_MODE
#include "tools/Debug.hpp"
#endif

#ifdef TIMING_MODE
#include "tools/ScopedTimer.hpp"
#endif

#include <opencv2/opencv.hpp>

// /**
//  * @brief Struct to represent a bounding box.
//  */
// struct BoundingBox
// {
//     int x;
//     int y;
//     int width;
//     int height;

//     BoundingBox() : x(0), y(0), width(0), height(0) {}
//     BoundingBox(int x_, int y_, int width_, int height_)
//         : x(x_), y(y_), width(width_), height(height_) {}
// };

// /**
//  * @brief Struct to represent a detection.
//  */
// struct Detection
// {
//     BoundingBox box;
//     float conf{};
//     int classId{};
//     Detection(BoundingBox box, float conf, int classId)
//         : box(box), classId(classId), conf(conf) {}
// };

/**
 * @namespace utils
 * @brief Namespace containing utility functions for the YOLO11Detector.
 */
namespace utils
{

    class MathUtils
    {
    public:
        /**
         * @brief A robust implementation of a clamp function.
         *        Restricts a value to lie within a specified range [low, high].
         *
         * @tparam T The type of the value to clamp. Should be an arithmetic type (int, float, etc.).
         * @param value The value to clamp.
         * @param low The lower bound of the range.
         * @param high The upper bound of the range.
         * @return const T& The clamped value, constrained to the range [low, high].
         *
         * @note If low > high, the function swaps the bounds automatically to ensure valid behavior.
         */
        template <typename T>
        static typename std::enable_if<std::is_arithmetic<T>::value, T>::type inline clamp(const T &value, const T &low, const T &high);

        /**
         * @brief Computes the product of elements in a vector.
         *
         * @param vector Vector of integers.
         * @return size_t Product of all elements.
         */
        static size_t vectorProduct(const std::vector<int64_t> &vector);
    };

    class ImagePreprocessingUtils
    {
    public:


        static inline void letterBox_old(const cv::Mat& image, cv::Mat& outImage,
                        const cv::Size& newShape,
                        const cv::Scalar& color,
                        bool auto_,
                        bool scaleFill,
                        bool scaleUp,
                        int stride);
        /**
         * @brief Resizes an image with letterboxing to maintain aspect ratio.
         *
         * @param image Input image.
         * @param outImage Output resized and padded image.
         * @param newShape Desired output size, default is (640, 640).
         * @param auto_ If True, use minimum rectangle to resize. If False, use new_shape directly.
         * @param scaleFill Whether to scale to fill the new shape without keeping aspect ratio.
         * @param scaleUp Whether to allow scaling up of the image.
         * @param center If True, center the placed image. If False, place image in top-left corner.
         * @param stride Stride of the model (e.g., 32 for YOLOv5).
         * @param padding_value Padding value (default is 114). 
         * @param interpolation Interpolation method (default is cv::INTER_LINEAR).
         */
        static void letterBox(const cv::Mat &image, cv::Mat &outImage,
                                     const cv::Size &newShape = cv::Size(640, 640),
                                     bool auto_ = false,
                                     bool scaleFill = false,
                                     bool scaleUp = true,
                                     bool center = true,
                                     int stride = 32,
                                     const cv::Scalar &padding_value = cv::Scalar(114, 114, 114),
                                     int interpolation = cv::INTER_LINEAR);
        

        /**
         * @brief Scales detection coordinates back to the original image size.
         *
         * @param imageShape Shape of the resized image used for inference.
         * @param bbox Detection bounding box to be scaled.
         * @param imageOriginalShape Original image size before resizing.
         * @param p_Clip Whether to clip the coordinates to the image boundaries.
         * @return BoundingBox Scaled bounding box.
         */
        static detectiondata::BoundingBox scaleCoords(const cv::Size &imageShape, detectiondata::BoundingBox coords,
                                       const cv::Size &imageOriginalShape, bool p_Clip);
    };

    class DrawingUtils
    {
    public:
        /**
         * @brief Generates a vector of colors for each class name.
         *
         * @param classNames Vector of class names.
         * @param seed Seed for random color generation to ensure reproducibility.
         * @return std::vector<cv::Scalar> Vector of colors.
         */
        static std::vector<cv::Scalar> generateColors(const std::vector<std::string> &classNames, int seed = 42);

        /**
         * @brief Draws bounding boxes and labels on the image based on detections.
         *
         * @param image Image on which to draw.
         * @param detections Vector of detections.
         * @param classNames Vector of class names corresponding to object IDs.
         * @param colors Vector of colors for each class.
         */
        static void drawBoundingBox(cv::Mat &image, const std::vector<detectiondata::Detection> &detections,
                                           const std::vector<std::string> &classNames, const std::vector<cv::Scalar> &colors, const float confidence_threshold = 0.0);

        /**
         * @brief Draws bounding boxes and semi-transparent masks on the image based on detections.
         *
         * @param image Image on which to draw.
         * @param detections Vector of detections.
         * @param classNames Vector of class names corresponding to object IDs.
         * @param classColors Vector of colors for each class.
         * @param maskAlpha Alpha value for the mask transparency.
         */
        static void drawBoundingBoxMask(cv::Mat &image, const std::vector<detectiondata::Detection> &detections,
                                               const std::vector<std::string> &classNames, const std::vector<cv::Scalar> &classColors,
                                               float maskAlpha = 0.4f, const float confidence_threshold = 0.0);
    };

    /**
     * @brief Loads class names from a given file path.
     *
     * @param path Path to the file containing class names.
     * @return std::vector<std::string> Vector of class names.
     */
    std::vector<std::string> getClassNames(const std::string &path);

    /**
     * @brief Performs Non-Maximum Suppression (NMS) on the bounding boxes.
     *
     * @param boundingBoxes Vector of bounding boxes.
     * @param scores Vector of confidence scores corresponding to each bounding box.
     * @param scoreThreshold Confidence threshold to filter boxes.
     * @param nmsThreshold IoU threshold for NMS.
     * @param indices Output vector of indices that survive NMS.
     */
    // Optimized Non-Maximum Suppression Function
    void NMSBoxes(const std::vector<detectiondata::BoundingBox> &boundingBoxes,
                  const std::vector<float> &scores,
                  float scoreThreshold,
                  float nmsThreshold,
                  std::vector<int> &indices);

};


/**
 * @brief YOLODetector class handles loading the YOLO model, preprocessing images, running inference, and postprocessing results.
 */
class YOLODetector {
public:
    /**
     * @brief Constructor to initialize the YOLO detector with model and label paths.
     * 
     * @param modelPath Path to the ONNX model file.
     * @param labelsPath Path to the file containing class labels.
     * @param useGPU Whether to use GPU for inference (default is false).
     */
    YOLODetector(const std::string &modelPath, const std::string &labelsPath, bool useGPU = false);
    
    /**
     * @brief Runs detection on the provided image.
     * 
     * @param image Input image for detection.
     * @param confThreshold Confidence threshold to filter detections (default is 0.4).
     * @param iouThreshold IoU threshold for Non-Maximum Suppression (default is 0.45).
     * @return std::vector<Detection> Vector of detections.
     */
    bool detect(const cv::Mat &image, std::vector<detectiondata::Detection> &detections, float confThreshold = 0.4f, float iouThreshold = 0.45f);

    /**
     * @brief Runs detection on a batch of images.
     * 
     * @param images Vector of input images for detection.
     * @param confThreshold Confidence threshold to filter detections (default is 0.4).
     * @param iouThreshold IoU threshold for Non-Maximum Suppression (default is 0.45).
     * @return std::vector<std::vector<Detection>> Vector of detections for each image.
     */
    std::vector<std::vector<detectiondata::Detection>> detect(const std::vector<cv::Mat> &images, float confThreshold = 0.4f, float iouThreshold = 0.45f);

    
    /**
     * @brief Draws bounding boxes on the image based on detections.
     * 
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     */
    void drawBoundingBox(cv::Mat &image, const std::vector<detectiondata::Detection> &detections) const {
        utils::DrawingUtils::drawBoundingBox(image, detections, classNames, classColors);
    }
    
    /**
     * @brief Draws bounding boxes and semi-transparent masks on the image based on detections.
     * 
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     * @param maskAlpha Alpha value for mask transparency (default is 0.4).
     */
    void drawBoundingBoxMask(cv::Mat &image, const std::vector<detectiondata::Detection> &detections, float maskAlpha = 0.4f) const {
        utils::DrawingUtils::drawBoundingBoxMask(image, detections, classNames, classColors, maskAlpha);
    }

    /**
     * @brief Gets the device used for inference.
     * 
     * @return std::string "GPU" or "CPU".
     */
    std::string getDevice() const { return device_used; }

private:
    Ort::Env env{nullptr};                         // ONNX Runtime environment
    Ort::SessionOptions sessionOptions{nullptr};   // Session options for ONNX Runtime
    Ort::Session session{nullptr};                 // ONNX Runtime session for running inference
    bool isDynamicInputShape{};                    // Flag indicating if input shape is dynamic
    bool isDynamicBatchSize{};                     // Flag indicating if batch size is dynamic
    cv::Size inputImageShape; 
    
    #if ONNXRUNTIME_VERSION <= 11
        std::vector<const char *> inputNodeNameAllocatedStrings;
        std::vector<const char *> outputNodeNameAllocatedStrings;
    #else
        std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
        std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    #endif
    
    // Expected input image shape for the model
    // Vectors to hold allocated input and output node names
    // std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char *> inputNames;
    // std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char *> outputNames;

    size_t numInputNodes, numOutputNodes;          // Number of input and output nodes in the model

    std::vector<std::string> classNames;            // Vector of class names loaded from file
    std::vector<cv::Scalar> classColors;            // Vector of colors for each class
    std::string device_used;                        // Device used for inference: "GPU" or "CPU"

    /**
     * @brief Preprocesses the input image for model inference.
     * 
     * @param image Input image.
     * @param blob Reference to pointer where preprocessed data will be stored.
     * @param inputTensorShape Reference to vector representing input tensor shape.
     * @return cv::Mat Resized image after preprocessing.
     */
    cv::Mat preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);

    /**
     * @brief Preprocesses a batch of images for model inference.
     * 
     * @param images Vector of input images.
     * @param blob Reference to pointer where preprocessed batch data will be stored.
     * @param inputTensorShape Reference to vector representing input tensor shape.
     * @return std::vector<cv::Size> Vector of resized image shapes.
     */
    std::vector<cv::Size> batch_preprocess(const std::vector<cv::Mat> &images, float *&blob, std::vector<int64_t> &inputTensorShape);
    
    /**
     * @brief Postprocesses the model output to extract detections.
     * 
     * @param originalImageSize Size of the original input image.
     * @param resizedImageShape Size of the image after preprocessing.
     * @param outputTensors Vector of output tensors from the model.
     * @param img_idx Index of the image we need to process its output
     * @param confThreshold Confidence threshold to filter detections.
     * @param iouThreshold IoU threshold for Non-Maximum Suppression.
     * @return std::vector<Detection> Vector of detections.
     */
    bool postprocess(const cv::Size &originalImageSize, const cv::Size &resizedImageShape,
         std::vector<detectiondata::Detection> &detections,
                                      const std::vector<Ort::Value> &outputTensors,int img_idx,
                                      float confThreshold, float iouThreshold);
    /**
     * @brief Postprocesses the model output to extract detections.
     * 
     * @param originalImageSize Size of the original input image.
     * @param resizedImageShape Size of the image after preprocessing.
     * @param outputTensors Vector of output tensors from the model.
     * @param img_idx Index of the image we need to process its output
     * @param confThreshold Confidence threshold to filter detections.
     * @param iouThreshold IoU threshold for Non-Maximum Suppression.
     * @return std::vector<Detection> Vector of detections.
     */
    bool postprocess_yolo10(const cv::Size &originalImageSize, const cv::Size &resizedImageShape, std::vector<detectiondata::Detection> &detections,
                                      const std::vector<Ort::Value> &outputTensors,int img_idx,
                                      float confThreshold, float iouThreshold);
    /**
     * @brief Postprocesses the model output to extract detections.
     * 
     * @param originalImageSize Size of the original input image.
     * @param resizedImageShape Size of the image after preprocessing.
     * @param outputTensors Vector of output tensors from the model.
     * @param img_idx Index of the image we need to process its output
     * @param confThreshold Confidence threshold to filter detections.
     * @param iouThreshold IoU threshold for Non-Maximum Suppression.
     * @return std::vector<Detection> Vector of detections.
     */
    bool postprocess_yolo7(const cv::Size &originalImageSize, const cv::Size &resizedImageShape, std::vector<detectiondata::Detection> &detections,
                                      const std::vector<Ort::Value> &outputTensors,int img_idx,
                                      float confThreshold, float iouThreshold);
    /**
     * @brief Postprocesses the model output to extract detections.
     * 
     * @param originalImageSize Size of the original input image.
     * @param resizedImageShape Size of the image after preprocessing.
     * @param outputTensors Vector of output tensors from the model.
     * @param img_idx Index of the image we need to process its output
     * @param confThreshold Confidence threshold to filter detections.
     * @param iouThreshold IoU threshold for Non-Maximum Suppression.
     * @return std::vector<Detection> Vector of detections.
     */
    bool postprocess_yolonas(
                const cv::Size &originalImageSize,
                const cv::Size &resizedImageShape,
                std::vector<detectiondata::Detection> &detections,
                const std::vector<Ort::Value> &outputTensors,int img_idx,
                float confThreshold,
                float iouThreshold
                );
    
};