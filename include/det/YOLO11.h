#pragma once

/**
 * @file YOLO11Detector.hpp
 * @brief Header file for the YOLO11Detector class, responsible for object detection
 *        using the YOLOv11 model with optimized performance for minimal latency.
 */

// Include necessary ONNX Runtime and OpenCV headers
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

// Include debug and custom ScopedTimer tools for performance measurement
#include "detection.h"

/**
 * @brief Confidence threshold for filtering detections.
 */
const float CONFIDENCE_THRESHOLD = 0.25f;
/**
 * @brief  IoU threshold for filtering detections.
 */
const float IOU_THRESHOLD = 0.45f;

/**
 * @namespace yolo_utils
 * @brief Namespace containing utility functions for the YOLO11Detector.
 */
namespace yolo_utils {

    template <typename T>
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type
    inline clamp(const T &value, const T &low, const T &high)
    {
        T validLow = low < high ? low : high;
        T validHigh = low < high ? high : low;
        if (value < validLow)
            return validLow;
        if (value > validHigh)
            return validHigh;
        return value;
    }

    std::vector<std::string> getClassNames(const std::string &path);
    size_t vectorProduct(const std::vector<int64_t> &vector);

    inline void letterBox(const cv::Mat& image, cv::Mat& outImage,
                        const cv::Size& newShape,
                        const cv::Scalar& color = cv::Scalar(114, 114, 114),
                        bool auto_ = true,
                        bool scaleFill = false,
                        bool scaleUp = true,
                        int stride = 32);
    
    BoundingBox scaleCoords(const cv::Size &imageShape, BoundingBox coords,
                            const cv::Size &imageOriginalShape, bool p_Clip);

    void NMSBoxes(const std::vector<BoundingBox>& boundingBoxes,
                const std::vector<float>& scores,
                float scoreThreshold,
                float nmsThreshold,
                std::vector<int>& indices);

    inline std::vector<cv::Scalar> generateColors(const std::vector<std::string> &classNames, int seed = 42);

    void drawBoundingBox(cv::Mat &image, const std::vector<Detection> &detections,
                                const std::vector<std::string> &classNames, const std::vector<cv::Scalar> &colors);

    void drawBoundingBoxMask(cv::Mat &image, const std::vector<Detection> &detections,
                                    const std::vector<std::string> &classNames, const std::vector<cv::Scalar> &classColors,
                                    float maskAlpha = 0.4f);
}

/**
 * @brief YOLO11Detector class handles loading the YOLO model, preprocessing images, running inference, and postprocessing results.
 */
 class YOLO11Detector {
    public:
        /**
         * @brief Constructor to initialize the YOLO detector with model and label paths.
         * 
         * @param modelPath Path to the ONNX model file.
         * @param labelsPath Path to the file containing class labels.
         * @param useGPU Whether to use GPU for inference (default is false).
         */
        YOLO11Detector(const std::string &modelPath, const std::string &labelsPath, bool useGPU = false);
        
        /**
         * @brief Runs detection on the provided image.
         * 
         * @param image Input image for detection.
         * @param confThreshold Confidence threshold to filter detections (default is 0.4).
         * @param iouThreshold IoU threshold for Non-Maximum Suppression (default is 0.45).
         * @return std::vector<Detection> Vector of detections.
         */
        bool detect(const cv::Mat &image, std::vector<Detection> &detections, float confThreshold = 0.4f, float iouThreshold = 0.45f);
        
        /**
         * @brief Draws bounding boxes on the image based on detections.
         * 
         * @param image Image on which to draw.
         * @param detections Vector of detections.
         */
        void drawBoundingBox(cv::Mat &image, const std::vector<Detection> &detections) const {
            yolo_utils::drawBoundingBox(image, detections, classNames, classColors);
        }
        
        /**
         * @brief Draws bounding boxes and semi-transparent masks on the image based on detections.
         * 
         * @param image Image on which to draw.
         * @param detections Vector of detections.
         * @param maskAlpha Alpha value for mask transparency (default is 0.4).
         */
        void drawBoundingBoxMask(cv::Mat &image, const std::vector<Detection> &detections, float maskAlpha = 0.4f) const {
            yolo_utils::drawBoundingBoxMask(image, detections, classNames, classColors, maskAlpha);
        }
    
    private:
        Ort::Env env{nullptr};                         // ONNX Runtime environment
        Ort::SessionOptions sessionOptions{nullptr};   // Session options for ONNX Runtime
        Ort::Session session{nullptr};                 // ONNX Runtime session for running inference
        bool isDynamicInputShape{};                    // Flag indicating if input shape is dynamic
        cv::Size inputImageShape;                      // Expected input image shape for the model
    
        // Vectors to hold allocated input and output node names
        std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
        std::vector<const char *> inputNames;
        std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
        std::vector<const char *> outputNames;
    
        size_t numInputNodes, numOutputNodes;          // Number of input and output nodes in the model
    
        std::vector<std::string> classNames;            // Vector of class names loaded from file
        std::vector<cv::Scalar> classColors;            // Vector of colors for each class
    
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
         * @brief Postprocesses the model output to extract detections.
         * 
         * @param originalImageSize Size of the original input image.
         * @param resizedImageShape Size of the image after preprocessing.
         * @param outputTensors Vector of output tensors from the model.
         * @param confThreshold Confidence threshold to filter detections.
         * @param iouThreshold IoU threshold for Non-Maximum Suppression.
         * @return std::vector<Detection> Vector of detections.
         */
        bool postprocess(const cv::Size &originalImageSize, const cv::Size &resizedImageShape, std::vector<Detection> &detections,
                                          const std::vector<Ort::Value> &outputTensors,
                                          float confThreshold, float iouThreshold);
        
    };