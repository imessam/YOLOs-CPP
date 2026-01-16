#pragma once

// ===================================
// Single YOLO-OBB Detector Header File
// ===================================
//
// This header defines the YOLOOBBDetector class for performing object
// detection using the YOLO-OBB model.

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "utils/YOLO-Utils.hpp"

const float CONFIDENCE_THRESHOLD = 0.25f;

// Using standardized structs from YOLO-Utils.hpp
// OrientedBoundingBox is already defined in YOLO-Utils.hpp
// OBB_Detection is already defined in YOLO-Utils.hpp

/**
 * @namespace OBB_NMS
 * @brief Namespace containing FIXED NMS functions using STANDARD ROTATED IOU
 */
namespace OBB_NMS {

float computeRotatedIoU(const OrientedBoundingBox &box1,
                        const OrientedBoundingBox &box2);

std::vector<int> nmsRotated(const std::vector<OrientedBoundingBox> &boxes,
                            const std::vector<float> &scores,
                            float iou_threshold = 0.45f);

std::vector<OBB_Detection>
nonMaxSuppression(const std::vector<OBB_Detection> &detections,
                  float iou_threshold = 0.45f, int max_det = 300);

} // namespace OBB_NMS

namespace utils {

/**
 * @brief Draws oriented bounding boxes on an image with labels and confidence
 * scores.
 */
void drawBoundingBox(cv::Mat &image,
                     const std::vector<OBB_Detection> &detections,
                     const std::vector<std::string> &classNames,
                     const std::vector<cv::Scalar> &classColors);

} // namespace utils

/**
 * @brief YOLO-OBB-Detector class handles loading the YOLO model, preprocessing
 * images, running inference, and postprocessing results.
 */
class YOLOOBBDetector {
public:
  YOLOOBBDetector(const std::string &modelPath, const std::string &labelsPath,
                  bool useGPU = false);

  std::vector<OBB_Detection> detect(const cv::Mat &image,
                                    float confThreshold = 0.25f,
                                    float iouThreshold = 0.25);

  void drawBoundingBox(cv::Mat &image,
                       const std::vector<OBB_Detection> &detections) const {
    utils::drawBoundingBox(image, detections, classNames, classColors);
  }

private:
  Ort::Env env{nullptr};
  Ort::SessionOptions sessionOptions{nullptr};
  Ort::Session session{nullptr};
  bool isDynamicInputShape{};
  cv::Size inputImageShape;

  std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
  std::vector<const char *> inputNames;
  std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
  std::vector<const char *> outputNames;

  size_t numInputNodes, numOutputNodes;

  std::vector<std::string> classNames;
  std::vector<cv::Scalar> classColors;

  cv::Mat preprocess(const cv::Mat &image, float *&blob,
                     std::vector<int64_t> &inputTensorShape);

  std::vector<OBB_Detection>
  postprocess(const cv::Size &originalImageSize,
              const cv::Size &resizedImageShape,
              const std::vector<Ort::Value> &outputTensors, float confThreshold,
              float iouThreshold, int topk = 500);
};
