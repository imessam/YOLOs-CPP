#pragma once

// ============================================================================
// YOLO Instance Segmentation
// ============================================================================
// Instance segmentation using YOLO models with mask prediction.
// Supports YOLOv8-seg and YOLOv11-seg models.
//
// Authors:
// YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// 2- Mohamed Samir, www.linkedin.com/in/mohamed-samir-7a730b237/
// ============================================================================

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "yolos/core/preprocessing.hpp"
#include "yolos/core/session_base.hpp"
#include "yolos/core/types.hpp"

namespace yolos {
namespace seg {

// ============================================================================
// Segmentation Result Structure
// ============================================================================

/// @brief Segmentation result containing bounding box, confidence, class ID,
/// and mask
struct Segmentation {
  BoundingBox box;  ///< Axis-aligned bounding box
  float conf{0.0f}; ///< Confidence score
  int classId{0};   ///< Class ID
  cv::Mat mask;     ///< Binary mask (CV_8UC1) in original image coordinates

  Segmentation() = default;
  Segmentation(const BoundingBox &box_, float conf_, int classId_,
               const cv::Mat &mask_)
      : box(box_), conf(conf_), classId(classId_), mask(mask_) {}
};

// ============================================================================
// YOLOSegDetector Class
// ============================================================================

/// @brief YOLO segmentation detector with mask prediction
class YOLOSegDetector : public OrtSessionBase {
public:
  /// @brief Constructor
  /// @param modelPath Path to the ONNX model file
  /// @param labelsPath Path to the class names file
  /// @param useGPU Whether to use GPU for inference
  YOLOSegDetector(const std::string &modelPath, const std::string &labelsPath,
                  bool useGPU = false);

  virtual ~YOLOSegDetector() = default;

  /// @brief Run segmentation on an image (optimized with buffer reuse)
  /// @param image Input image (BGR format)
  /// @param confThreshold Confidence threshold
  /// @param iouThreshold IoU threshold for NMS
  /// @return Vector of segmentation results
  std::vector<Segmentation> segment(const cv::Mat &image,
                                    float confThreshold = 0.4f,
                                    float iouThreshold = 0.45f);

  /// @brief Draw segmentations with boxes and labels on an image
  /// @param image Image to draw on
  /// @param results Vector of segmentation results
  /// @param maskAlpha Mask transparency (0-1)
  void drawSegmentations(cv::Mat &image,
                         const std::vector<Segmentation> &results,
                         float maskAlpha = 0.5f) const;

  /// @brief Draw only segmentation masks (no boxes)
  void drawMasksOnly(cv::Mat &image, const std::vector<Segmentation> &results,
                     float maskAlpha = 0.5f) const;

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
  static constexpr float MASK_THRESHOLD = 0.5f;

  // Pre-allocated buffer for inference (avoids per-frame allocations)
  mutable preprocessing::InferenceBuffer buffer_;

  /// @brief Postprocess segmentation outputs
  virtual std::vector<Segmentation>
  postprocess(const cv::Size &originalSize, const cv::Size &letterboxSize,
              const std::vector<Ort::Value> &outputTensors, float confThreshold,
              float iouThreshold);

  /// @brief Postprocess YOLO26-seg format outputs (end-to-end, no NMS needed)
  virtual std::vector<Segmentation>
  postprocessV26(const cv::Size &originalSize, const cv::Size &letterboxSize,
                 const float *output0, const float *output1,
                 const std::vector<int64_t> &shape0,
                 const std::vector<int64_t> &shape1, float confThreshold);
};

} // namespace seg
} // namespace yolos
