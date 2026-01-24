#pragma once

// ============================================================================
// YOLO Preprocessing Utilities
// ============================================================================
// Optimized image preprocessing functions for YOLO inference including
// letterbox resizing, coordinate scaling, and blob conversion.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <opencv2/opencv.hpp>
#include <vector>

#include "yolos/core/types.hpp"

namespace yolos {
namespace preprocessing {

// ============================================================================
// Pre-allocated Buffer for Inference
// ============================================================================

/// @brief Pre-allocated inference buffer to avoid per-frame allocations
struct InferenceBuffer {
  std::vector<float> blob; ///< CHW format blob for ONNX
  cv::Mat resized;         ///< Letterboxed image
  cv::Mat rgbFloat;        ///< RGB float image
  cv::Size lastInputSize;  ///< Last input size (for reuse check)
  cv::Size lastTargetSize; ///< Last target size

  /// @brief Ensure blob has required capacity
  void ensureCapacity(int height, int width, int channels = 3);
};

// ============================================================================
// LetterBox Resizing
// ============================================================================

/// @brief Resize an image with letterboxing to maintain aspect ratio
/// @param image Input image
/// @param outImage Output resized and padded image
/// @param newShape Desired output size
/// @param color Padding color (default is gray 114,114,114)
/// @param autoSize If true, use minimum rectangle to resize
/// @param scaleFill Whether to scale to fill without keeping aspect ratio
/// @param scaleUp Whether to allow scaling up of the image
/// @param stride Stride size for padding alignment
void letterBox(const cv::Mat &image, cv::Mat &outImage,
               const cv::Size &newShape,
               const cv::Scalar &color = cv::Scalar(114, 114, 114),
               bool autoSize = true, bool scaleFill = false,
               bool scaleUp = true, int stride = 32);

/// @brief Alternative letterbox with center option (matches Ultralytics)
/// @param image Input image
/// @param outImage Output resized and padded image
/// @param newShape Desired output size (default 640x640)
/// @param autoSize If true, use minimum rectangle to resize
/// @param scaleFill Whether to scale to fill without keeping aspect ratio
/// @param scaleUp Whether to allow scaling up of the image
/// @param center If true, center the placed image
/// @param stride Stride of the model
/// @param paddingValue Padding value (default is 114)
/// @param interpolation Interpolation method
void letterBoxCentered(const cv::Mat &image, cv::Mat &outImage,
                       const cv::Size &newShape = cv::Size(640, 640),
                       bool autoSize = false, bool scaleFill = false,
                       bool scaleUp = true, bool center = true, int stride = 32,
                       const cv::Scalar &paddingValue = cv::Scalar(114, 114,
                                                                   114),
                       int interpolation = cv::INTER_LINEAR);

// ============================================================================
// Coordinate Scaling
// ============================================================================

/// @brief Scale detection coordinates from letterbox space back to original
/// image size
/// @param letterboxShape Shape of the letterboxed image used for inference
/// @param coords Bounding box in letterbox coordinates
/// @param originalShape Original image size before letterboxing
/// @param clip Whether to clip coordinates to image boundaries
/// @return Scaled bounding box in original image coordinates
BoundingBox scaleCoords(const cv::Size &letterboxShape,
                        const BoundingBox &coords,
                        const cv::Size &originalShape, bool clip = true);

/// @brief Scale keypoint coordinates from letterbox space back to original
/// image size
/// @param letterboxShape Shape of the letterboxed image
/// @param keypoint Keypoint in letterbox coordinates
/// @param originalShape Original image size
/// @param clip Whether to clip coordinates to image boundaries
/// @return Scaled keypoint in original image coordinates
KeyPoint scaleKeypoint(const cv::Size &letterboxShape, const KeyPoint &keypoint,
                       const cv::Size &originalShape, bool clip = true);

/// @brief Get letterbox padding and scale parameters
/// @param originalShape Original image size
/// @param letterboxShape Letterboxed image size
/// @param[out] scale Scale factor applied
/// @param[out] padX Horizontal padding
/// @param[out] padY Vertical padding
void getLetterboxParams(const cv::Size &originalShape,
                        const cv::Size &letterboxShape, float &scale,
                        float &padX, float &padY);

// ============================================================================
// Optimized Single-Pass Preprocessing
// ============================================================================

/// @brief Fast letterbox with direct blob output (avoids intermediate copies)
/// @param image Input BGR image
/// @param blob Output CHW float blob (pre-allocated)
/// @param targetSize Target size for inference
/// @param[out] actualSize Actual output size after letterboxing
/// @param padColor Padding color value (0-255, default 114)
void letterBoxToBlob(const cv::Mat &image, std::vector<float> &blob,
                     const cv::Size &targetSize, cv::Size &actualSize,
                     float padColor = 114.0f);

/// @brief Fast letterbox with buffer reuse
/// @param image Input BGR image
/// @param buffer Pre-allocated inference buffer
/// @param targetSize Target size for inference
/// @param[out] actualSize Actual output size
/// @param dynamicShape Whether to use dynamic shape
void letterBoxToBlob(const cv::Mat &image, InferenceBuffer &buffer,
                     const cv::Size &targetSize, cv::Size &actualSize,
                     bool dynamicShape = false);

/// @brief Get scale and padding info from letterbox operation
/// @param originalSize Original image size
/// @param letterboxSize Letterboxed image size
/// @param[out] scale Scale factor
/// @param[out] padX X padding
/// @param[out] padY Y padding
void getScalePad(const cv::Size &originalSize, const cv::Size &letterboxSize,
                 float &scale, float &padX, float &padY);

/// @brief Fast coordinate descaling (batch operation)
/// @param coords Array of x,y coordinates to descale
/// @param count Number of coordinate pairs
/// @param scale Letterbox scale
/// @param padX X padding
/// @param padY Y padding
void descaleCoordsBatch(float *coords, size_t count, float scale, float padX,
                        float padY);

} // namespace preprocessing
} // namespace yolos
