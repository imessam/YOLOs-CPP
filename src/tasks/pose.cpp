#include "yolos/tasks/pose.hpp"
#include "yolos/core/drawing.hpp"
#include "yolos/core/nms.hpp"
#include "yolos/core/utils.hpp"
#include <cmath>
#include <iostream>

namespace yolos {
namespace pose {

YOLOPoseDetector::YOLOPoseDetector(const std::string &modelPath,
                                   const std::string &labelsPath, bool useGPU)
    : OrtSessionBase(modelPath, useGPU) {

  if (!labelsPath.empty()) {
    classNames_ = utils::getClassNames(labelsPath);
  } else {
    classNames_ = {"person"};
  }
  classColors_ = drawing::generateColors(classNames_);

  // Pre-allocate inference buffer
  buffer_.ensureCapacity(inputShape_.height, inputShape_.width, 3);
}

std::vector<PoseResult> YOLOPoseDetector::detect(const cv::Mat &image,
                                                 float confThreshold,
                                                 float iouThreshold) {
  // Optimized preprocessing with buffer reuse
  cv::Size actualSize;
  preprocessing::letterBoxToBlob(image, buffer_, inputShape_, actualSize,
                                 isDynamicInputShape());

  // Create input tensor (uses pre-allocated blob)
  std::vector<int64_t> inputTensorShape = {1, 3, actualSize.height,
                                           actualSize.width};
  Ort::Value inputTensor =
      createInputTensor(buffer_.blob.data(), inputTensorShape);

  // Run inference
  std::vector<Ort::Value> outputTensors = runInference(inputTensor);

  // Postprocess
  return postprocess(image.size(), actualSize, outputTensors, confThreshold,
                     iouThreshold);
}

void YOLOPoseDetector::drawPoses(cv::Mat &image,
                                 const std::vector<PoseResult> &results,
                                 int kptRadius, float kptThreshold,
                                 int lineThickness) const {
  for (const auto &pose : results) {
    // Draw bounding box
    cv::rectangle(
        image, cv::Point(pose.box.x, pose.box.y),
        cv::Point(pose.box.x + pose.box.width, pose.box.y + pose.box.height),
        cv::Scalar(0, 255, 0), lineThickness);

    // Draw keypoints and skeleton
    drawing::drawPoseSkeleton(image, pose.keypoints, getPoseSkeleton(),
                              kptRadius, kptThreshold, lineThickness);
  }
}

void YOLOPoseDetector::drawSkeletonsOnly(cv::Mat &image,
                                         const std::vector<PoseResult> &results,
                                         int kptRadius, float kptThreshold,
                                         int lineThickness) const {
  for (const auto &pose : results) {
    drawing::drawPoseSkeleton(image, pose.keypoints, getPoseSkeleton(),
                              kptRadius, kptThreshold, lineThickness);
  }
}

std::vector<PoseResult>
YOLOPoseDetector::postprocess(const cv::Size &originalSize,
                              const cv::Size &resizedShape,
                              const std::vector<Ort::Value> &outputTensors,
                              float confThreshold, float iouThreshold) {
  const float *rawOutput = outputTensors[0].GetTensorData<float>();
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

  // Detect output format based on shape:
  // YOLOv8/v11: [1, 56, num_detections] - requires NMS
  // YOLO26:     [1, 300, 57] - end-to-end, NMS-free
  const int expectedFeaturesV8 =
      4 + 1 + NUM_KEYPOINTS * FEATURES_PER_KEYPOINT; // 56
  const int expectedFeaturesV26 =
      4 + 1 + 1 + NUM_KEYPOINTS * FEATURES_PER_KEYPOINT; // 57

  if (outputShape.size() == 3 && outputShape[2] == expectedFeaturesV26) {
    return postprocessV26(originalSize, resizedShape, rawOutput, outputShape,
                          confThreshold);
  } else if (outputShape.size() == 3 && outputShape[1] == expectedFeaturesV8) {
    return postprocessV8(originalSize, resizedShape, rawOutput, outputShape,
                         confThreshold, iouThreshold);
  } else {
    std::cerr << "[ERROR] Unsupported pose model output shape: ["
              << outputShape[0] << ", " << outputShape[1] << ", "
              << outputShape[2] << "]" << std::endl;
    return {};
  }
}

std::vector<PoseResult> YOLOPoseDetector::postprocessV8(
    const cv::Size &originalSize, const cv::Size &resizedShape,
    const float *rawOutput, const std::vector<int64_t> &outputShape,
    float confThreshold, float iouThreshold) {
  std::vector<PoseResult> results;
  const size_t numDetections = outputShape[2];

  // Pre-compute scale and padding
  float scale, padX, padY;
  preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
  const float invScale = 1.0f / scale;

  std::vector<BoundingBox> boxes;
  std::vector<float> confidences;
  std::vector<std::vector<KeyPoint>> allKeypoints;
  boxes.reserve(64);
  confidences.reserve(64);
  allKeypoints.reserve(64);

  for (size_t d = 0; d < numDetections; ++d) {
    const float objConfidence = rawOutput[4 * numDetections + d];
    if (objConfidence < confThreshold)
      continue;

    // Decode bounding box (cx, cy, w, h format)
    const float cx = rawOutput[0 * numDetections + d];
    const float cy = rawOutput[1 * numDetections + d];
    const float w = rawOutput[2 * numDetections + d];
    const float h = rawOutput[3 * numDetections + d];

    // Convert to original image coordinates
    BoundingBox box;
    box.x = utils::clamp(static_cast<int>((cx - w * 0.5f - padX) * invScale), 0,
                         originalSize.width - 1);
    box.y = utils::clamp(static_cast<int>((cy - h * 0.5f - padY) * invScale), 0,
                         originalSize.height - 1);
    box.width = utils::clamp(static_cast<int>(w * invScale), 1,
                             originalSize.width - box.x);
    box.height = utils::clamp(static_cast<int>(h * invScale), 1,
                              originalSize.height - box.y);

    // Extract keypoints
    std::vector<KeyPoint> keypoints;
    keypoints.reserve(NUM_KEYPOINTS);
    for (int k = 0; k < NUM_KEYPOINTS; ++k) {
      const int offset = 5 + k * FEATURES_PER_KEYPOINT;
      KeyPoint kpt;
      kpt.x = (rawOutput[offset * numDetections + d] - padX) * invScale;
      kpt.y = (rawOutput[(offset + 1) * numDetections + d] - padY) * invScale;
      const float rawConf = rawOutput[(offset + 2) * numDetections + d];
      kpt.confidence = 1.0f / (1.0f + std::exp(-rawConf)); // Sigmoid

      kpt.x =
          utils::clamp(kpt.x, 0.0f, static_cast<float>(originalSize.width - 1));
      kpt.y = utils::clamp(kpt.y, 0.0f,
                           static_cast<float>(originalSize.height - 1));

      keypoints.push_back(kpt);
    }

    boxes.push_back(box);
    confidences.push_back(objConfidence);
    allKeypoints.push_back(std::move(keypoints));
  }

  if (boxes.empty())
    return results;

  std::vector<int> indices;
  nms::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);

  results.reserve(indices.size());
  for (int idx : indices) {
    results.emplace_back(boxes[idx], confidences[idx], 0, allKeypoints[idx]);
  }

  return results;
}

std::vector<PoseResult> YOLOPoseDetector::postprocessV26(
    const cv::Size &originalSize, const cv::Size &resizedShape,
    const float *rawOutput, const std::vector<int64_t> &outputShape,
    float confThreshold) {
  std::vector<PoseResult> results;
  const size_t numDetections = outputShape[1];
  const size_t numFeatures = outputShape[2];

  float scale, padX, padY;
  preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
  const float invScale = 1.0f / scale;

  for (size_t d = 0; d < numDetections; ++d) {
    const size_t base = d * numFeatures;
    const float conf = rawOutput[base + 4];
    if (conf < confThreshold)
      continue;

    const float x1 = rawOutput[base + 0];
    const float y1 = rawOutput[base + 1];
    const float x2 = rawOutput[base + 2];
    const float y2 = rawOutput[base + 3];

    BoundingBox box;
    box.x = utils::clamp(static_cast<int>((x1 - padX) * invScale), 0,
                         originalSize.width - 1);
    box.y = utils::clamp(static_cast<int>((y1 - padY) * invScale), 0,
                         originalSize.height - 1);
    const int x2_scaled = utils::clamp(static_cast<int>((x2 - padX) * invScale),
                                       0, originalSize.width - 1);
    const int y2_scaled = utils::clamp(static_cast<int>((y2 - padY) * invScale),
                                       0, originalSize.height - 1);
    box.width = std::max(1, x2_scaled - box.x);
    box.height = std::max(1, y2_scaled - box.y);

    std::vector<KeyPoint> keypoints;
    keypoints.reserve(NUM_KEYPOINTS);
    for (int k = 0; k < NUM_KEYPOINTS; ++k) {
      const size_t kptBase = base + 6 + k * FEATURES_PER_KEYPOINT;
      KeyPoint kpt;
      kpt.x = (rawOutput[kptBase + 0] - padX) * invScale;
      kpt.y = (rawOutput[kptBase + 1] - padY) * invScale;
      kpt.confidence = rawOutput[kptBase + 2];

      kpt.x =
          utils::clamp(kpt.x, 0.0f, static_cast<float>(originalSize.width - 1));
      kpt.y = utils::clamp(kpt.y, 0.0f,
                           static_cast<float>(originalSize.height - 1));

      keypoints.push_back(kpt);
    }

    results.emplace_back(box, conf, 0, std::move(keypoints));
  }

  return results;
}

} // namespace pose
} // namespace yolos
