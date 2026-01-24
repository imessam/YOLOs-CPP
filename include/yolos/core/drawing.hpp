#pragma once

// ============================================================================
// YOLO Drawing Utilities
// ============================================================================
// Visualization functions for detection results including bounding boxes,
// labels, masks, and pose skeletons.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "yolos/core/types.hpp"

namespace yolos {
namespace drawing {

// ============================================================================
// Color Generation
// ============================================================================

/// @brief Generate consistent random colors for each class
/// @param classNames Vector of class names
/// @param seed Random seed for reproducibility
/// @return Vector of BGR colors
std::vector<cv::Scalar>
generateColors(const std::vector<std::string> &classNames, int seed = 42);

/// @brief Get the Ultralytics pose palette colors
/// @return Vector of BGR colors for pose visualization
const std::vector<cv::Scalar> &getPosePalette();

// ============================================================================
// Bounding Box Drawing
// ============================================================================

/// @brief Draw a single bounding box with label on an image
/// @param image Image to draw on
/// @param box Bounding box
/// @param label Text label
/// @param color Box color
/// @param thickness Line thickness
void drawBoundingBox(cv::Mat &image, const BoundingBox &box,
                     const std::string &label, const cv::Scalar &color,
                     int thickness = 2);

/// @brief Draw a bounding box with semi-transparent mask fill
/// @param image Image to draw on
/// @param box Bounding box
/// @param label Text label
/// @param color Box color
/// @param maskAlpha Transparency of the mask fill (0-1)
void drawBoundingBoxWithMask(cv::Mat &image, const BoundingBox &box,
                             const std::string &label, const cv::Scalar &color,
                             float maskAlpha = 0.4f);

// ============================================================================
// Oriented Bounding Box Drawing
// ============================================================================

/// @brief Draw an oriented bounding box on an image
/// @param image Image to draw on
/// @param obb Oriented bounding box
/// @param label Text label
/// @param color Box color
/// @param thickness Line thickness
void drawOrientedBoundingBox(cv::Mat &image, const OrientedBoundingBox &obb,
                             const std::string &label, const cv::Scalar &color,
                             int thickness = 2);

// ============================================================================
// Pose Drawing
// ============================================================================

/// @brief Draw pose keypoints and skeleton on an image
/// @param image Image to draw on
/// @param keypoints Vector of keypoints
/// @param skeleton Skeleton connections
/// @param kptRadius Keypoint circle radius
/// @param kptThreshold Minimum confidence to draw keypoint
/// @param lineThickness Skeleton line thickness
void drawPoseSkeleton(cv::Mat &image, const std::vector<KeyPoint> &keypoints,
                      const std::vector<std::pair<int, int>> &skeleton,
                      int kptRadius = 4, float kptThreshold = 0.5f,
                      int lineThickness = 2);

// ============================================================================
// Segmentation Mask Drawing
// ============================================================================

/// @brief Draw a segmentation mask on an image
/// @param image Image to draw on
/// @param mask Binary mask (CV_8UC1)
/// @param color Mask color
/// @param alpha Mask transparency (0-1)
void drawSegmentationMask(cv::Mat &image, const cv::Mat &mask,
                          const cv::Scalar &color, float alpha = 0.5f);

} // namespace drawing
} // namespace yolos
