#pragma once

// ============================================================================
// YOLO Image Classification
// ============================================================================
// Image classification using YOLO models (v11, v12, YOLO26).
// Supports efficient classification with Ultralytics-style preprocessing.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "yolos/core/preprocessing.hpp"
#include "yolos/core/session_base.hpp"
#include "yolos/core/version.hpp"

namespace yolos {
namespace cls {

// ============================================================================
// Classification Result Structure
// ============================================================================

/// @brief Classification result containing class ID, confidence, and class name
struct ClassificationResult {
  int classId{-1};         ///< Predicted class ID
  float confidence{0.0f};  ///< Confidence score
  std::string className{}; ///< Human-readable class name

  ClassificationResult() = default;
  ClassificationResult(int id, float conf, std::string name)
      : classId(id), confidence(conf), className(std::move(name)) {}
};

// ============================================================================
// Drawing Utility for Classification
// ============================================================================

/// @brief Draw classification result on an image
void drawClassificationResult(cv::Mat &image,
                              const ClassificationResult &result,
                              const cv::Point &position = cv::Point(10, 30),
                              const cv::Scalar &textColor = cv::Scalar(0, 255,
                                                                       0),
                              const cv::Scalar &bgColor = cv::Scalar(0, 0, 0));

// ============================================================================
// YOLOClassifier Class
// ============================================================================

/// @brief YOLO classifier for image classification
class YOLOClassifier : public OrtSessionBase {
public:
  /// @brief Constructor
  /// @param modelPath Path to the ONNX model file
  /// @param labelsPath Path to the class names file
  /// @param useGPU Whether to use GPU for inference
  YOLOClassifier(const std::string &modelPath, const std::string &labelsPath,
                 bool useGPU = false);

  virtual ~YOLOClassifier() = default;

  /// @brief Run classification on an image
  /// @param image Input image (BGR format)
  /// @return Classification result
  ClassificationResult classify(const cv::Mat &image);

  /// @brief Draw classification result on an image
  void drawResult(cv::Mat &image, const ClassificationResult &result,
                  const cv::Point &position = cv::Point(10, 30)) const;

  /// @brief Get class names
  [[nodiscard]] const std::vector<std::string> &getClassNames() const {
    return classNames_;
  }

protected:
  std::vector<std::string> classNames_;

  // Pre-allocated buffer for inference
  mutable preprocessing::InferenceBuffer buffer_;

  /// @brief Preprocess image for classification (Ultralytics-style)
  void preprocess(const cv::Mat &image, std::vector<int64_t> &inputTensorShape);

  /// @brief Postprocess classification output
  ClassificationResult
  postprocess(const std::vector<Ort::Value> &outputTensors);
};

// ============================================================================
// Version-Specific Classifier Subclasses
// ============================================================================

/// @brief YOLOv11 classifier
class YOLO11Classifier : public YOLOClassifier {
public:
  YOLO11Classifier(const std::string &modelPath, const std::string &labelsPath,
                   bool useGPU = false)
      : YOLOClassifier(modelPath, labelsPath, useGPU) {}
};

/// @brief YOLOv12 classifier
class YOLO12Classifier : public YOLOClassifier {
public:
  YOLO12Classifier(const std::string &modelPath, const std::string &labelsPath,
                   bool useGPU = false)
      : YOLOClassifier(modelPath, labelsPath, useGPU) {}
};

/// @brief YOLO26 classifier
class YOLO26Classifier : public YOLOClassifier {
public:
  YOLO26Classifier(const std::string &modelPath, const std::string &labelsPath,
                   bool useGPU = false)
      : YOLOClassifier(modelPath, labelsPath, useGPU) {}
};

// ============================================================================
// Factory Function
// ============================================================================

/// @brief Create a classifier with explicit version selection
/// @param modelPath Path to the ONNX model
/// @param labelsPath Path to the class names file
/// @param version YOLO version
/// @param useGPU Whether to use GPU
/// @return Unique pointer to classifier
inline std::unique_ptr<YOLOClassifier>
createClassifier(const std::string &modelPath, const std::string &labelsPath,
                 yolos::YOLOVersion version = yolos::YOLOVersion::V11,
                 bool useGPU = false) {
  switch (version) {
  case yolos::YOLOVersion::V26:
    return std::make_unique<YOLO26Classifier>(modelPath, labelsPath, useGPU);
  case yolos::YOLOVersion::V12:
    return std::make_unique<YOLO12Classifier>(modelPath, labelsPath, useGPU);
  default:
    return std::make_unique<YOLO11Classifier>(modelPath, labelsPath, useGPU);
  }
}

} // namespace cls
} // namespace yolos
