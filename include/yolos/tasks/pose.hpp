#pragma once

// ============================================================================
// YOLO Pose Estimation
// ============================================================================
// Human pose estimation using YOLO models with keypoint detection.
// Supports YOLOv8-pose, YOLOv11-pose, and YOLO26-pose models.
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
namespace pose {

// ============================================================================
// Pose Result Structure
// ============================================================================

/// @brief Pose estimation result containing bounding box, confidence, and
/// keypoints
struct PoseResult {
  BoundingBox box;                 ///< Bounding box around the person
  float conf{0.0f};                ///< Detection confidence
  int classId{0};                  ///< Class ID (typically 0 for person)
  std::vector<KeyPoint> keypoints; ///< Detected keypoints (17 for COCO format)

  PoseResult() = default;
  PoseResult(const BoundingBox &box_, float conf_, int classId_,
             const std::vector<KeyPoint> &kpts)
      : box(box_), conf(conf_), classId(classId_), keypoints(kpts) {}
};

// ============================================================================
// YOLOPoseDetector Class
// ============================================================================

/// @brief YOLO pose estimation detector with keypoint detection
class YOLOPoseDetector : public OrtSessionBase {
public:
  /// @brief Constructor
  /// @param modelPath Path to the ONNX model file
  /// @param labelsPath Path to the class names file (optional for pose)
  /// @param useGPU Whether to use GPU for inference
  YOLOPoseDetector(const std::string &modelPath,
                   const std::string &labelsPath = "", bool useGPU = false);

  virtual ~YOLOPoseDetector() = default;

  /// @brief Run pose detection on an image (optimized with buffer reuse)
  /// @param image Input image (BGR format)
  /// @param confThreshold Confidence threshold
  /// @param iouThreshold IoU threshold for NMS
  /// @return Vector of pose results
  std::vector<PoseResult> detect(const cv::Mat &image,
                                 float confThreshold = 0.4f,
                                 float iouThreshold = 0.5f);

  /// @brief Draw pose estimations on an image
  /// @param image Image to draw on
  /// @param results Vector of pose results
  /// @param kptRadius Keypoint circle radius
  /// @param kptThreshold Minimum confidence to draw keypoint
  /// @param lineThickness Skeleton line thickness
  void drawPoses(cv::Mat &image, const std::vector<PoseResult> &results,
                 int kptRadius = 4, float kptThreshold = 0.5f,
                 int lineThickness = 2) const;

  /// @brief Draw only skeletons (no bounding boxes)
  void drawSkeletonsOnly(cv::Mat &image, const std::vector<PoseResult> &results,
                         int kptRadius = 4, float kptThreshold = 0.5f,
                         int lineThickness = 2) const;

  /// @brief Get class names
  [[nodiscard]] const std::vector<std::string> &getClassNames() const {
    return classNames_;
  }

  /// @brief Get COCO pose skeleton connections
  [[nodiscard]] static const std::vector<std::pair<int, int>> &
  getPoseSkeleton() {
    static const std::vector<std::pair<int, int>> skeleton = {
        {0, 1},   {0, 2},   {1, 3},   {2, 4},   // Face
        {3, 5},   {4, 6},                       // Head to shoulders
        {5, 7},   {7, 9},   {6, 8},   {8, 10},  // Arms
        {5, 6},   {5, 11},  {6, 12},  {11, 12}, // Body
        {11, 13}, {13, 15}, {12, 14}, {14, 16}  // Legs
    };
    return skeleton;
  }

protected:
  std::vector<std::string> classNames_;
  std::vector<cv::Scalar> classColors_;
  static constexpr int NUM_KEYPOINTS = 17;
  static constexpr int FEATURES_PER_KEYPOINT = 3;

  // Pre-allocated buffer for inference
  mutable preprocessing::InferenceBuffer buffer_;

  /// @brief Postprocess pose detection outputs
  virtual std::vector<PoseResult>
  postprocess(const cv::Size &originalSize, const cv::Size &resizedShape,
              const std::vector<Ort::Value> &outputTensors, float confThreshold,
              float iouThreshold);

  /// @brief Postprocess YOLOv8/v11 pose detection outputs (requires NMS)
  virtual std::vector<PoseResult>
  postprocessV8(const cv::Size &originalSize, const cv::Size &resizedShape,
                const float *rawOutput, const std::vector<int64_t> &outputShape,
                float confThreshold, float iouThreshold);

  /// @brief Postprocess YOLO26 pose detection outputs (end-to-end, NMS-free)
  virtual std::vector<PoseResult>
  postprocessV26(const cv::Size &originalSize, const cv::Size &resizedShape,
                 const float *rawOutput,
                 const std::vector<int64_t> &outputShape, float confThreshold);
};

} // namespace pose
} // namespace yolos
