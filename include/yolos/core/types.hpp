#pragma once

// ============================================================================
// YOLO Core Types
// ============================================================================
// Single source of truth for shared data structures used across all YOLO tasks.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <utility>
#include <vector>

namespace yolos {

// ============================================================================
// BoundingBox - Axis-aligned bounding box for detection, segmentation, pose
// ============================================================================
struct BoundingBox {
  int x{0};      ///< X-coordinate of top-left corner
  int y{0};      ///< Y-coordinate of top-left corner
  int width{0};  ///< Width of the bounding box
  int height{0}; ///< Height of the bounding box

  BoundingBox() = default;

  BoundingBox(int x_, int y_, int width_, int height_)
      : x(x_), y(y_), width(width_), height(height_) {}

  /// @brief Compute area of the bounding box
  [[nodiscard]] float area() const noexcept;

  /// @brief Compute intersection with another bounding box
  [[nodiscard]] BoundingBox intersect(const BoundingBox &other) const noexcept;

  /// @brief Compute IoU (Intersection over Union) with another bounding box
  [[nodiscard]] float iou(const BoundingBox &other) const noexcept;
};

// ============================================================================
// OrientedBoundingBox - Rotated bounding box for OBB detection
// ============================================================================
struct OrientedBoundingBox {
  float x{0.0f};      ///< X-coordinate of center
  float y{0.0f};      ///< Y-coordinate of center
  float width{0.0f};  ///< Width of the box
  float height{0.0f}; ///< Height of the box
  float angle{0.0f};  ///< Rotation angle in radians

  OrientedBoundingBox() = default;

  OrientedBoundingBox(float x_, float y_, float width_, float height_,
                      float angle_)
      : x(x_), y(y_), width(width_), height(height_), angle(angle_) {}

  /// @brief Compute area of the oriented bounding box
  [[nodiscard]] float area() const noexcept;
};

// ============================================================================
// KeyPoint - Single keypoint for pose estimation
// ============================================================================
struct KeyPoint {
  float x{0.0f};          ///< X-coordinate
  float y{0.0f};          ///< Y-coordinate
  float confidence{0.0f}; ///< Confidence score

  KeyPoint() = default;

  KeyPoint(float x_, float y_, float conf_ = 0.0f)
      : x(x_), y(y_), confidence(conf_) {}
};

// ============================================================================
// Skeleton connections for COCO pose format (17 keypoints)
// ============================================================================
const std::vector<std::pair<int, int>> &getPoseSkeleton();

} // namespace yolos
