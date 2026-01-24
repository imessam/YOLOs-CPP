#pragma once

// ============================================================================
// YOLO Object Detection
// ============================================================================
// Object detection using YOLO models with support for multiple versions
// (v7, v8, v10, v11, NAS) through runtime auto-detection or explicit selection.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <cfloat>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "yolos/core/preprocessing.hpp"
#include "yolos/core/session_base.hpp"
#include "yolos/core/types.hpp"
#include "yolos/core/version.hpp"

namespace yolos {
namespace det {

// ============================================================================
// Detection Result Structure
// ============================================================================

/// @brief Detection result containing bounding box, confidence, and class ID
struct Detection {
  BoundingBox box;  ///< Axis-aligned bounding box
  float conf{0.0f}; ///< Confidence score
  int classId{-1};  ///< Class ID

  Detection() = default;
  Detection(const BoundingBox &box_, float conf_, int classId_)
      : box(box_), conf(conf_), classId(classId_) {}
};

// ============================================================================
// YOLODetector Base Class
// ============================================================================

/// @brief Base YOLO detector with runtime version auto-detection
class YOLODetector : public OrtSessionBase {
public:
  /// @brief Constructor
  /// @param modelPath Path to the ONNX model file
  /// @param labelsPath Path to the class names file
  /// @param useGPU Whether to use GPU for inference
  /// @param version YOLO version (Auto for runtime detection)
  YOLODetector(const std::string &modelPath, const std::string &labelsPath,
               bool useGPU = false,
               yolos::YOLOVersion version = yolos::YOLOVersion::Auto);

  virtual ~YOLODetector() = default;

  /// @brief Run detection on an image (optimized with buffer reuse)
  /// @param image Input image (BGR format)
  /// @param confThreshold Confidence threshold
  /// @param iouThreshold IoU threshold for NMS
  /// @return Vector of detections
  virtual std::vector<Detection> detect(const cv::Mat &image,
                                        float confThreshold = 0.4f,
                                        float iouThreshold = 0.45f);

  /// @brief Draw detections on an image
  /// @param image Image to draw on
  /// @param detections Vector of detections
  void drawDetections(cv::Mat &image,
                      const std::vector<Detection> &detections) const;

  /// @brief Draw detections with semi-transparent mask fill
  void drawDetectionsWithMask(cv::Mat &image,
                              const std::vector<Detection> &detections,
                              float alpha = 0.4f) const;

  /// @brief Get class names
  [[nodiscard]] const std::vector<std::string> &getClassNames() const {
    return classNames_;
  }

  /// @brief Get class colors
  [[nodiscard]] const std::vector<cv::Scalar> &getClassColors() const {
    return classColors_;
  }

protected:
  YOLOVersion version_{YOLOVersion::Auto};
  std::vector<std::string> classNames_;
  std::vector<cv::Scalar> classColors_;

  // Pre-allocated buffer for inference (avoids per-frame allocations)
  mutable preprocessing::InferenceBuffer buffer_;

  /// @brief Detect YOLO version from output tensors
  YOLOVersion detectVersion(const std::vector<Ort::Value> &outputTensors);

  /// @brief Postprocess based on detected version
  virtual std::vector<Detection>
  postprocess(const cv::Size &originalSize, const cv::Size &resizedShape,
              const std::vector<Ort::Value> &outputTensors, YOLOVersion version,
              float confThreshold, float iouThreshold);

  /// @brief Standard postprocess for YOLOv8/v11 format [batch, features, boxes]
  /// Optimized: single box storage with batched NMS
  virtual std::vector<Detection>
  postprocessStandard(const cv::Size &originalSize,
                      const cv::Size &resizedShape,
                      const std::vector<Ort::Value> &outputTensors,
                      float confThreshold, float iouThreshold);

  /// @brief Postprocess for YOLOv7 format [batch, boxes, features]
  virtual std::vector<Detection>
  postprocessV7(const cv::Size &originalSize, const cv::Size &resizedShape,
                const std::vector<Ort::Value> &outputTensors,
                float confThreshold, float iouThreshold);

  /// @brief Postprocess for YOLOv10 format [batch, boxes, 6] (end-to-end, no
  /// NMS needed)
  virtual std::vector<Detection>
  postprocessV10(const cv::Size &originalSize, const cv::Size &resizedShape,
                 const std::vector<Ort::Value> &outputTensors,
                 float confThreshold, float iouThreshold);

  /// @brief Postprocess for YOLO-NAS format (two outputs: boxes and scores)
  virtual std::vector<Detection>
  postprocessNAS(const cv::Size &originalSize, const cv::Size &resizedShape,
                 const std::vector<Ort::Value> &outputTensors,
                 float confThreshold, float iouThreshold);
};

// ============================================================================
// Version-Specific Detector Subclasses
// ============================================================================

/// @brief YOLOv7 detector (forces V7 postprocessing)
class YOLOv7Detector : public YOLODetector {
public:
  YOLOv7Detector(const std::string &modelPath, const std::string &labelsPath,
                 bool useGPU = false)
      : YOLODetector(modelPath, labelsPath, useGPU, YOLOVersion::V7) {}
};

/// @brief YOLOv8 detector (forces standard postprocessing)
class YOLOv8Detector : public YOLODetector {
public:
  YOLOv8Detector(const std::string &modelPath, const std::string &labelsPath,
                 bool useGPU = false)
      : YOLODetector(modelPath, labelsPath, useGPU, YOLOVersion::V8) {}
};

/// @brief YOLOv10 detector (forces V10 end-to-end postprocessing)
class YOLOv10Detector : public YOLODetector {
public:
  YOLOv10Detector(const std::string &modelPath, const std::string &labelsPath,
                  bool useGPU = false)
      : YOLODetector(modelPath, labelsPath, useGPU, YOLOVersion::V10) {}
};

/// @brief YOLOv11 detector (forces standard postprocessing)
class YOLOv11Detector : public YOLODetector {
public:
  YOLOv11Detector(const std::string &modelPath, const std::string &labelsPath,
                  bool useGPU = false)
      : YOLODetector(modelPath, labelsPath, useGPU, YOLOVersion::V11) {}
};

/// @brief YOLO-NAS detector (forces NAS postprocessing)
class YOLONASDetector : public YOLODetector {
public:
  YOLONASDetector(const std::string &modelPath, const std::string &labelsPath,
                  bool useGPU = false)
      : YOLODetector(modelPath, labelsPath, useGPU, YOLOVersion::NAS) {}
};

/// @brief YOLOv26 detector (forces V26 end-to-end postprocessing)
class YOLO26Detector : public YOLODetector {
public:
  YOLO26Detector(const std::string &modelPath, const std::string &labelsPath,
                 bool useGPU = false)
      : YOLODetector(modelPath, labelsPath, useGPU, YOLOVersion::V26) {}
};

// ============================================================================
// Factory Function
// ============================================================================

/// @brief Create a detector with explicit version selection
/// @param modelPath Path to the ONNX model
/// @param labelsPath Path to the class names file
/// @param version YOLO version (Auto for runtime detection)
/// @param useGPU Whether to use GPU
/// @return Unique pointer to detector
inline std::unique_ptr<YOLODetector>
createDetector(const std::string &modelPath, const std::string &labelsPath,
               YOLOVersion version = YOLOVersion::Auto, bool useGPU = false) {
  switch (version) {
  case YOLOVersion::V7:
    return std::make_unique<YOLOv7Detector>(modelPath, labelsPath, useGPU);
  case YOLOVersion::V8:
    return std::make_unique<YOLOv8Detector>(modelPath, labelsPath, useGPU);
  case YOLOVersion::V10:
    return std::make_unique<YOLOv10Detector>(modelPath, labelsPath, useGPU);
  case YOLOVersion::V11:
    return std::make_unique<YOLOv11Detector>(modelPath, labelsPath, useGPU);
  case YOLOVersion::V26:
    return std::make_unique<YOLO26Detector>(modelPath, labelsPath, useGPU);
  case YOLOVersion::NAS:
    return std::make_unique<YOLONASDetector>(modelPath, labelsPath, useGPU);
  default:
    return std::make_unique<YOLODetector>(modelPath, labelsPath, useGPU,
                                          YOLOVersion::Auto);
  }
}

} // namespace det
} // namespace yolos
