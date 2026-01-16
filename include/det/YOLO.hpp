#pragma once

// ===================================
// Single YOLOv11 Detector Header File
// ===================================
//
// This header defines the YOLODetector class for performing object detection
// using the YOLOv11 model.

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

// Include debug and custom ScopedTimer tools for performance measurement
#ifdef DEBUG_MODE
#include "tools/Debug.hpp"
#endif

#ifdef TIMING_MODE
#include "tools/ScopedTimer.hpp"
#endif

#include "utils/YOLO-Utils.hpp"

/**
 * @brief YOLODetector class handles loading the YOLO model, preprocessing
 * images, running inference, and postprocessing results.
 */
class YOLODetector {
public:
  /**
   * @brief Constructor to initialize the YOLO detector with model and label
   * paths.
   */
  YOLODetector(const std::string &modelPath, const std::string &labelsPath,
               bool useGPU = false);

  /**
   * @brief Runs detection on the provided image.
   */
  std::vector<Detection> detect(const cv::Mat &image,
                                float confThreshold = 0.4f,
                                float iouThreshold = 0.45f);

  /**
   * @brief Runs detection on a batch of images.
   */
  std::vector<std::vector<Detection>> detect(const std::vector<cv::Mat> &images,
                                             float confThreshold = 0.4f,
                                             float iouThreshold = 0.45f);

  /**
   * @brief Draws bounding boxes on the image based on detections.
   */
  void drawBoundingBox(cv::Mat &image,
                       const std::vector<Detection> &detections) const {
    utils::DrawingUtils::drawBoundingBox(image, detections, classNames,
                                         classColors);
  }

  /**
   * @brief Draws bounding boxes and semi-transparent masks on the image based
   * on detections.
   */
  void drawBoundingBoxMask(cv::Mat &image,
                           const std::vector<Detection> &detections,
                           float maskAlpha = 0.4f) const {
    utils::DrawingUtils::drawBoundingBox(image, detections, classNames,
                                         classColors, maskAlpha);
  }

  /**
   * @brief Gets the device used for inference.
   */
  std::string getDevice() const { return device_used; }

private:
  Ort::Env env{nullptr}; // ONNX Runtime environment
  Ort::SessionOptions sessionOptions{
      nullptr};                  // Session options for ONNX Runtime
  Ort::Session session{nullptr}; // ONNX Runtime session for running inference
  bool isDynamicInputShape{};    // Flag indicating if input shape is dynamic
  bool isDynamicBatchSize{};     // Flag indicating if batch size is dynamic
  cv::Size inputImageShape;      // Expected input image shape for the model

  std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
  std::vector<const char *> inputNames;
  std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
  std::vector<const char *> outputNames;

  size_t numInputNodes,
      numOutputNodes; // Number of input and output nodes in the model

  std::vector<std::string> classNames; // Vector of class names loaded from file
  std::vector<cv::Scalar> classColors; // Vector of colors for each class
  std::string device_used; // Device used for inference: "GPU" or "CPU"

  /**
   * @brief Preprocesses the input image for model inference.
   */
  cv::Mat preprocess(const cv::Mat &image, float *&blob,
                     std::vector<int64_t> &inputTensorShape);

  /**
   * @brief Preprocesses a batch of images for model inference.
   */
  std::vector<cv::Size>
  batch_preprocess(const std::vector<cv::Mat> &images, float *&blob,
                   std::vector<int64_t> &inputTensorShape);

  /**
   * @brief Postprocesses the model output to extract detections.
   */
  std::vector<Detection>
  postprocess(const cv::Size &originalImageSize,
              const cv::Size &resizedImageShape,
              const std::vector<Ort::Value> &outputTensors, int img_idx,
              float confThreshold, float iouThreshold);

  std::vector<Detection>
  postprocess_yolo10(const cv::Size &originalImageSize,
                     const cv::Size &resizedImageShape,
                     const std::vector<Ort::Value> &outputTensors, int img_idx,
                     float confThreshold, float iouThreshold);

  std::vector<Detection>
  postprocess_yolo7(const cv::Size &originalImageSize,
                    const cv::Size &resizedImageShape,
                    const std::vector<Ort::Value> &outputTensors, int img_idx,
                    float confThreshold, float iouThreshold);

  std::vector<Detection>
  postprocess_yolonas(const cv::Size &originalImageSize,
                      const cv::Size &resizedImageShape,
                      const std::vector<Ort::Value> &outputTensors, int img_idx,
                      float confThreshold, float iouThreshold);
};
