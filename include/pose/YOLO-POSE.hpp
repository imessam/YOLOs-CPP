#pragma once

// ===================================
// Single YOLO Pose Detector Header File
// ===================================
//
// This header defines the YOLOPOSEDetector class for performing human pose
// estimation using the YOLO model.

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "utils/YOLO-Utils.hpp"

/**
 * @brief List of COCO skeleton connections for human pose estimation.
 */
const std::vector<std::pair<int, int>> POSE_SKELETON = {
    // Face connections
    {0, 1},
    {0, 2},
    {1, 3},
    {2, 4},
    // Head-to-shoulder connections
    {3, 5},
    {4, 6},
    // Arms
    {5, 7},
    {7, 9},
    {6, 8},
    {8, 10},
    // Body
    {5, 6},
    {5, 11},
    {6, 12},
    {11, 12},
    // Legs
    {11, 13},
    {13, 15},
    {12, 14},
    {14, 16}};

namespace utils {

/**
 * @brief Draws pose estimations including bounding boxes, keypoints, and
 * skeleton
 */
void drawPoseEstimation(cv::Mat &image,
                        const std::vector<Detection> &detections,
                        float confidenceThreshold = 0.5,
                        float kptThreshold = 0.5);

} // namespace utils

/**
 * @brief YOLOPOSEDetector class handles loading the YOLO model, preprocessing
 * images, running inference, and postprocessing results.
 */
class YOLOPOSEDetector {
public:
  YOLOPOSEDetector(const std::string &modelPath, const std::string &labelsPath,
                   bool useGPU = false);

  std::vector<Detection> detect(const cv::Mat &image,
                                float confThreshold = 0.4f,
                                float iouThreshold = 0.45f);

  void drawDetections(cv::Mat &image, const std::vector<Detection> &detections,
                      float confidenceThreshold = 0.5,
                      float kptThreshold = 0.5) const {
    utils::drawPoseEstimation(image, detections, confidenceThreshold,
                              kptThreshold);
  }

  std::string getDevice() const { return device_used; }

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

  std::string device_used;
  std::vector<std::string> classNames;
  std::vector<cv::Scalar> classColors;

  cv::Mat preprocess(const cv::Mat &image, float *&blob,
                     std::vector<int64_t> &inputTensorShape);

  std::vector<Detection>
  postprocess(const cv::Size &letterboxShape, const cv::Size &imageShape,
              const std::vector<Ort::Value> &outputTensors, float confThreshold,
              float iouThreshold);
};
