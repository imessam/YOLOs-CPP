#pragma once

// ====================================================
// Single YOLO Segmentation Header File
// ====================================================
//
// This header defines the YOLOSegDetector class for performing object detection
// and segmentation using the YOLO model.

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "utils/YOLO-Utils.hpp"

#ifdef DEBUG_MODE
#include "tools/Debug.hpp"
#endif

#ifdef TIMING_MODE
#include "tools/ScopedTimer.hpp"
#endif
// ============================================================================
// Constants / Thresholds
// ============================================================================
static const float CONFIDENCE_THRESHOLD =
    0.40f;                                // Filter boxes below this confidence
static const float IOU_THRESHOLD = 0.45f; // NMS IoU threshold
static const float MASK_THRESHOLD =
    0.40f; // Slightly lower to capture partial objects

// Compatibility alias
using Segmentation = Detection;

// ============================================================================
// YOLOSegDetector Class
// ============================================================================
class YOLOSegDetector {
public:
  YOLOSegDetector(const std::string &modelPath, const std::string &labelsPath,
                  bool useGPU = false);

  // Main API
  std::vector<Segmentation> segment(const cv::Mat &image,
                                    float confThreshold = CONFIDENCE_THRESHOLD,
                                    float iouThreshold = IOU_THRESHOLD);

  // Draw results
  void drawSegmentationsAndBoxes(cv::Mat &image,
                                 const std::vector<Segmentation> &results,
                                 float maskAlpha = 0.5f) const;

  void drawSegmentations(cv::Mat &image,
                         const std::vector<Segmentation> &results,
                         float maskAlpha = 0.5f) const;
  // Accessors
  const std::vector<std::string> &getClassNames() const { return classNames; }
  const std::vector<cv::Scalar> &getClassColors() const { return classColors; }

private:
  Ort::Env env;
  Ort::SessionOptions sessionOptions;
  Ort::Session session{nullptr};

  bool isDynamicInputShape{false};
  cv::Size inputImageShape;

  std::vector<Ort::AllocatedStringPtr> inputNameAllocs;
  std::vector<const char *> inputNames;
  std::vector<Ort::AllocatedStringPtr> outputNameAllocs;
  std::vector<const char *> outputNames;

  size_t numInputNodes = 0;
  size_t numOutputNodes = 0;

  std::vector<std::string> classNames;
  std::vector<cv::Scalar> classColors;

  // Helpers
  cv::Mat preprocess(const cv::Mat &image, float *&blobPtr,
                     std::vector<int64_t> &inputTensorShape);

  std::vector<Segmentation> postprocess(const cv::Size &origSize,
                                        const cv::Size &letterboxSize,
                                        const std::vector<Ort::Value> &outputs,
                                        float confThreshold,
                                        float iouThreshold);
};
