#pragma once

// ============================================================================
// YOLO Non-Maximum Suppression
// ============================================================================
// NMS implementations for axis-aligned and oriented bounding boxes.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <vector>

#include <opencv2/opencv.hpp>

#include "yolos/core/types.hpp"

namespace yolos {
namespace nms {

// ============================================================================
// Standard NMS for Axis-Aligned Bounding Boxes
// ============================================================================

/// @brief Perform Non-Maximum Suppression on bounding boxes
/// @param boxes Vector of bounding boxes
/// @param scores Vector of confidence scores
/// @param scoreThreshold Minimum score to consider
/// @param nmsThreshold IoU threshold for suppression
/// @param[out] indices Output indices of boxes that survived NMS
void NMSBoxes(const std::vector<BoundingBox> &boxes,
              const std::vector<float> &scores, float scoreThreshold,
              float nmsThreshold, std::vector<int> &indices);

// ============================================================================
// Float-Precision NMS for Letterbox Coordinates
// ============================================================================

/// @brief Perform NMS on float-precision bounding boxes (for letterbox space)
/// @param boxes Vector of cv::Rect2f boxes
/// @param scores Vector of confidence scores
/// @param scoreThreshold Minimum score to consider
/// @param nmsThreshold IoU threshold for suppression
/// @param[out] indices Output indices of boxes that survived NMS
void NMSBoxesF(const std::vector<cv::Rect2f> &boxes,
               const std::vector<float> &scores, float scoreThreshold,
               float nmsThreshold, std::vector<int> &indices);

/// @brief Perform class-aware NMS on float-precision boxes
void NMSBoxesFBatched(const std::vector<cv::Rect2f> &boxes,
                      const std::vector<float> &scores,
                      const std::vector<int> &classIds, float scoreThreshold,
                      float nmsThreshold, std::vector<int> &indices);

// ============================================================================
// Rotated NMS for Oriented Bounding Boxes
// ============================================================================

/// @brief Compute IoU between two oriented bounding boxes using OpenCV
/// @param box1 First oriented bounding box
/// @param box2 Second oriented bounding box
/// @return IoU value between 0 and 1
float computeRotatedIoU(const OrientedBoundingBox &box1,
                        const OrientedBoundingBox &box2);

/// @brief Perform NMS on oriented bounding boxes using rotated IoU
/// @param boxes Vector of oriented bounding boxes
/// @param scores Vector of confidence scores
/// @param nmsThreshold IoU threshold for suppression
/// @param maxDet Maximum number of detections to keep
/// @return Indices of boxes that survived NMS
std::vector<int> NMSRotated(const std::vector<OrientedBoundingBox> &boxes,
                            const std::vector<float> &scores,
                            float nmsThreshold = 0.45f, int maxDet = 300);

// ============================================================================
// Batched NMS (per-class NMS)
// ============================================================================

/// @brief Perform class-aware NMS by offsetting boxes by class ID
/// @param boxes Vector of bounding boxes
/// @param scores Vector of confidence scores
/// @param classIds Vector of class IDs
/// @param scoreThreshold Minimum score to consider
/// @param nmsThreshold IoU threshold for suppression
/// @param[out] indices Output indices of boxes that survived NMS
void NMSBoxesBatched(const std::vector<BoundingBox> &boxes,
                     const std::vector<float> &scores,
                     const std::vector<int> &classIds, float scoreThreshold,
                     float nmsThreshold, std::vector<int> &indices);

/// @brief Perform class-aware NMS on oriented bounding boxes
/// @param boxes Vector of oriented bounding boxes
/// @param scores Vector of confidence scores
/// @param classIds Vector of class IDs
/// @param nmsThreshold IoU threshold for suppression
/// @param maxDet Maximum number of detections to keep
/// @return Indices of boxes that survived NMS
std::vector<int>
NMSRotatedBatched(const std::vector<OrientedBoundingBox> &boxes,
                  const std::vector<float> &scores,
                  const std::vector<int> &classIds, float nmsThreshold = 0.45f,
                  int maxDet = 300);

} // namespace nms
} // namespace yolos
