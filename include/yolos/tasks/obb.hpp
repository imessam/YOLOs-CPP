#pragma once

// ============================================================================
// YOLO Oriented Bounding Box Detection (OBB)
// ============================================================================
// Object detection with rotated/oriented bounding boxes for aerial imagery
// and other scenarios requiring rotation-aware detection.
// Supports YOLOv8-obb, YOLOv11-obb, and YOLO26-obb models.
//
// Authors:
// YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// 2- Mohamed Samir, www.linkedin.com/in/mohamed-samir-7a730b237/
// 3- Khaled Gabr, https://www.linkedin.com/in/khalidgabr/
// ============================================================================

#include <cfloat>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "yolos/core/preprocessing.hpp"
#include "yolos/core/session_base.hpp"
#include "yolos/core/types.hpp"

namespace yolos {
namespace obb {

// ============================================================================
// OBB Detection Result Structure
// ============================================================================

/// @brief OBB detection result containing oriented bounding box, confidence,
/// and class ID
struct OBBResult {
  OrientedBoundingBox box; ///< Oriented bounding box (center-based with angle)
  float conf{0.0f};        ///< Confidence score
  int classId{-1};         ///< Class ID

  OBBResult() = default;
  OBBResult(const OrientedBoundingBox &box_, float conf_, int classId_)
      : box(box_), conf(conf_), classId(classId_) {}
};

// ============================================================================
// YOLOOBBDetector Class
// ============================================================================

/// @brief YOLO oriented bounding box detector for rotated object detection
class YOLOOBBDetector : public OrtSessionBase {
public:
  /// @brief Constructor
  /// @param modelPath Path to the ONNX model file
  /// @param labelsPath Path to the class names file
  /// @param useGPU Whether to use GPU for inference
  YOLOOBBDetector(const std::string &modelPath, const std::string &labelsPath,
                  bool useGPU = false);

  virtual ~YOLOOBBDetector() = default;

  /// @brief Run OBB detection on an image (optimized with buffer reuse)
  /// @param image Input image (BGR format)
  /// @param confThreshold Confidence threshold
  /// @param iouThreshold IoU threshold for NMS
  /// @param maxDet Maximum number of detections to return
  /// @return Vector of OBB detection results
  std::vector<OBBResult> detect(const cv::Mat &image,
                                float confThreshold = 0.25f,
                                float iouThreshold = 0.45f, int maxDet = 300);

  /// @brief Draw OBB detections on an image
  /// @param image Image to draw on
  /// @param results Vector of OBB detection results
  /// @param thickness Line thickness
  void drawDetections(cv::Mat &image, const std::vector<OBBResult> &results,
                      int thickness = 2) const;

  /// @brief Get class names
  [[nodiscard]] const std::vector<std::string> &getClassNames() const {
    return classNames_;
  }

  /// @brief Get class colors
  [[nodiscard]] const std::vector<cv::Scalar> &getClassColors() const {
    return classColors_;
  }

protected:
  std::vector<std::string> classNames_;
  std::vector<cv::Scalar> classColors_;

  // Pre-allocated buffer for inference
  mutable preprocessing::InferenceBuffer buffer_;

  /// @brief Postprocess OBB detection outputs
  std::vector<OBBResult>
  postprocess(const cv::Size &originalSize, const cv::Size &resizedShape,
              const std::vector<Ort::Value> &outputTensors, float confThreshold,
              float iouThreshold, int maxDet);

  /// @brief Postprocess YOLOv8/v11 OBB detection outputs (requires NMS)
  std::vector<OBBResult>
  postprocessV8(const cv::Size &originalSize, const cv::Size &resizedShape,
                const float *rawOutput, const std::vector<int64_t> &outputShape,
                float confThreshold, float iouThreshold, int maxDet);

  /// @brief Postprocess YOLO26 OBB detection outputs (end-to-end, NMS-free)
  std::vector<OBBResult> postprocessV26(const cv::Size &originalSize,
                                        const cv::Size &resizedShape,
                                        const float *rawOutput,
                                        const std::vector<int64_t> &outputShape,
                                        float confThreshold, int maxDet);
};

} // namespace obb
} // namespace yolos
