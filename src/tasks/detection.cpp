#include "yolos/tasks/detection.hpp"
#include "yolos/core/drawing.hpp"
#include "yolos/core/nms.hpp"
#include "yolos/core/utils.hpp"
#include <cfloat>

namespace yolos {
namespace det {

YOLODetector::YOLODetector(const std::string &modelPath,
                           const std::string &labelsPath, bool useGPU,
                           YOLOVersion version)
    : OrtSessionBase(modelPath, useGPU), version_(version) {
  classNames_ = utils::getClassNames(labelsPath);
  classColors_ = drawing::generateColors(classNames_);

  // Pre-allocate inference buffer
  buffer_.ensureCapacity(inputShape_.height, inputShape_.width, 3);
}

std::vector<Detection> YOLODetector::detect(const cv::Mat &image,
                                            float confThreshold,
                                            float iouThreshold) {
  // Optimized preprocessing with buffer reuse
  cv::Size actualSize;
  preprocessing::letterBoxToBlob(image, buffer_, inputShape_, actualSize,
                                 isDynamicInputShape_);

  // Create input tensor (uses pre-allocated blob)
  std::vector<int64_t> inputTensorShape = {1, 3, actualSize.height,
                                           actualSize.width};
  Ort::Value inputTensor =
      createInputTensor(buffer_.blob.data(), inputTensorShape);

  // Run inference
  std::vector<Ort::Value> outputTensors = runInference(inputTensor);

  // Determine version if auto
  YOLOVersion effectiveVersion = version_;
  if (effectiveVersion == YOLOVersion::Auto) {
    effectiveVersion = detectVersion(outputTensors);
  }

  // Postprocess based on version
  return postprocess(image.size(), actualSize, outputTensors, effectiveVersion,
                     confThreshold, iouThreshold);
}

void YOLODetector::drawDetections(
    cv::Mat &image, const std::vector<Detection> &detections) const {
  for (const auto &det : detections) {
    if (det.classId >= 0 &&
        static_cast<size_t>(det.classId) < classNames_.size()) {
      std::string label = classNames_[det.classId] + ": " +
                          std::to_string(static_cast<int>(det.conf * 100)) +
                          "%";
      const cv::Scalar &color = classColors_[det.classId % classColors_.size()];
      drawing::drawBoundingBox(image, det.box, label, color);
    }
  }
}

void YOLODetector::drawDetectionsWithMask(
    cv::Mat &image, const std::vector<Detection> &detections,
    float alpha) const {
  for (const auto &det : detections) {
    if (det.classId >= 0 &&
        static_cast<size_t>(det.classId) < classNames_.size()) {
      std::string label = classNames_[det.classId] + ": " +
                          std::to_string(static_cast<int>(det.conf * 100)) +
                          "%";
      const cv::Scalar &color = classColors_[det.classId % classColors_.size()];
      drawing::drawBoundingBoxWithMask(image, det.box, label, color, alpha);
    }
  }
}

YOLOVersion
YOLODetector::detectVersion(const std::vector<Ort::Value> &outputTensors) {
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  return version::detectFromOutputShape(outputShape, outputTensors.size());
}

std::vector<Detection> YOLODetector::postprocess(
    const cv::Size &originalSize, const cv::Size &resizedShape,
    const std::vector<Ort::Value> &outputTensors, YOLOVersion version,
    float confThreshold, float iouThreshold) {
  switch (version) {
  case YOLOVersion::V7:
    return postprocessV7(originalSize, resizedShape, outputTensors,
                         confThreshold, iouThreshold);
  case YOLOVersion::V10:
  case YOLOVersion::V26:
    return postprocessV10(originalSize, resizedShape, outputTensors,
                          confThreshold, iouThreshold);
  case YOLOVersion::NAS:
    return postprocessNAS(originalSize, resizedShape, outputTensors,
                          confThreshold, iouThreshold);
  default:
    return postprocessStandard(originalSize, resizedShape, outputTensors,
                               confThreshold, iouThreshold);
  }
}

std::vector<Detection>
YOLODetector::postprocessStandard(const cv::Size &originalSize,
                                  const cv::Size &resizedShape,
                                  const std::vector<Ort::Value> &outputTensors,
                                  float confThreshold, float iouThreshold) {
  std::vector<Detection> detections;
  const float *rawOutput = outputTensors[0].GetTensorData<float>();
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

  const size_t numFeatures = outputShape[1];
  const size_t numDetections = outputShape[2];

  if (numDetections == 0)
    return detections;

  const int numClasses = static_cast<int>(numFeatures) - 4;
  if (numClasses <= 0)
    return detections;

  // Pre-compute scale and padding once
  float scale, padX, padY;
  preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
  const float invScale = 1.0f / scale;

  std::vector<BoundingBox> boxes;
  std::vector<float> confs;
  std::vector<int> classIds;
  boxes.reserve(256); // Reasonable initial capacity
  confs.reserve(256);
  classIds.reserve(256);

  for (size_t d = 0; d < numDetections; ++d) {
    // Find max class score
    int classId = 0;
    float maxScore = rawOutput[4 * numDetections + d];
    for (int c = 1; c < numClasses; ++c) {
      const float score = rawOutput[(4 + c) * numDetections + d];
      if (score > maxScore) {
        maxScore = score;
        classId = c;
      }
    }

    if (maxScore > confThreshold) {
      const float centerX = rawOutput[0 * numDetections + d];
      const float centerY = rawOutput[1 * numDetections + d];
      const float width = rawOutput[2 * numDetections + d];
      const float height = rawOutput[3 * numDetections + d];

      // Convert center to corner and descale in one step
      const float left = (centerX - width * 0.5f - padX) * invScale;
      const float top = (centerY - height * 0.5f - padY) * invScale;
      const float w = width * invScale;
      const float h = height * invScale;

      // Clip to image bounds
      BoundingBox box;
      box.x = utils::clamp(static_cast<int>(left), 0, originalSize.width - 1);
      box.y = utils::clamp(static_cast<int>(top), 0, originalSize.height - 1);
      box.width =
          utils::clamp(static_cast<int>(w), 1, originalSize.width - box.x);
      box.height =
          utils::clamp(static_cast<int>(h), 1, originalSize.height - box.y);

      boxes.push_back(box);
      confs.push_back(maxScore);
      classIds.push_back(classId);
    }
  }

  // Batched NMS (handles class offsets internally)
  std::vector<int> indices;
  nms::NMSBoxesBatched(boxes, confs, classIds, confThreshold, iouThreshold,
                       indices);

  detections.reserve(indices.size());
  for (int idx : indices) {
    detections.emplace_back(boxes[idx], confs[idx], classIds[idx]);
  }

  return detections;
}

std::vector<Detection>
YOLODetector::postprocessV7(const cv::Size &originalSize,
                            const cv::Size &resizedShape,
                            const std::vector<Ort::Value> &outputTensors,
                            float confThreshold, float iouThreshold) {
  std::vector<Detection> detections;
  const float *rawOutput = outputTensors[0].GetTensorData<float>();
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

  const size_t numDetections = outputShape[1];
  const size_t numFeatures = outputShape[2];

  if (numDetections == 0)
    return detections;

  const int numClasses = static_cast<int>(numFeatures) - 5;
  if (numClasses <= 0)
    return detections;

  // Pre-compute scale and padding
  float scale, padX, padY;
  preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
  const float invScale = 1.0f / scale;

  std::vector<BoundingBox> boxes;
  std::vector<float> confs;
  std::vector<int> classIds;
  boxes.reserve(256);
  confs.reserve(256);
  classIds.reserve(256);

  for (size_t d = 0; d < numDetections; ++d) {
    const float objConf = rawOutput[d * numFeatures + 4];
    if (objConf <= confThreshold)
      continue;

    const float centerX = rawOutput[d * numFeatures + 0];
    const float centerY = rawOutput[d * numFeatures + 1];
    const float width = rawOutput[d * numFeatures + 2];
    const float height = rawOutput[d * numFeatures + 3];

    int classId = 0;
    float maxScore = rawOutput[d * numFeatures + 5];
    for (int c = 1; c < numClasses; ++c) {
      const float score = rawOutput[d * numFeatures + 5 + c];
      if (score > maxScore) {
        maxScore = score;
        classId = c;
      }
    }

    // Convert and descale in one step
    const float left = (centerX - width * 0.5f - padX) * invScale;
    const float top = (centerY - height * 0.5f - padY) * invScale;

    BoundingBox box;
    box.x = utils::clamp(static_cast<int>(left), 0, originalSize.width - 1);
    box.y = utils::clamp(static_cast<int>(top), 0, originalSize.height - 1);
    box.width = utils::clamp(static_cast<int>(width * invScale), 1,
                             originalSize.width - box.x);
    box.height = utils::clamp(static_cast<int>(height * invScale), 1,
                              originalSize.height - box.y);

    boxes.push_back(box);
    confs.push_back(objConf);
    classIds.push_back(classId);
  }

  std::vector<int> indices;
  nms::NMSBoxesBatched(boxes, confs, classIds, confThreshold, iouThreshold,
                       indices);

  detections.reserve(indices.size());
  for (int idx : indices) {
    detections.emplace_back(boxes[idx], confs[idx], classIds[idx]);
  }

  return detections;
}

std::vector<Detection>
YOLODetector::postprocessV10(const cv::Size &originalSize,
                             const cv::Size &resizedShape,
                             const std::vector<Ort::Value> &outputTensors,
                             float confThreshold, float /*iouThreshold*/) {
  std::vector<Detection> detections;
  const float *rawOutput = outputTensors[0].GetTensorData<float>();
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

  const int numDetections = static_cast<int>(outputShape[1]);

  // Pre-compute scale and padding
  float scale, padX, padY;
  preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
  const float invScale = 1.0f / scale;

  detections.reserve(numDetections);

  for (int i = 0; i < numDetections; ++i) {
    const float confidence = rawOutput[i * 6 + 4];
    if (confidence <= confThreshold)
      continue;

    const float x1 = (rawOutput[i * 6 + 0] - padX) * invScale;
    const float y1 = (rawOutput[i * 6 + 1] - padY) * invScale;
    const float x2 = (rawOutput[i * 6 + 2] - padX) * invScale;
    const float y2 = (rawOutput[i * 6 + 3] - padY) * invScale;
    const int classId = static_cast<int>(rawOutput[i * 6 + 5]);

    BoundingBox box;
    box.x = utils::clamp(static_cast<int>(x1), 0, originalSize.width - 1);
    box.y = utils::clamp(static_cast<int>(y1), 0, originalSize.height - 1);
    box.width =
        utils::clamp(static_cast<int>(x2 - x1), 1, originalSize.width - box.x);
    box.height =
        utils::clamp(static_cast<int>(y2 - y1), 1, originalSize.height - box.y);

    detections.emplace_back(box, confidence, classId);
  }

  return detections;
}

std::vector<Detection>
YOLODetector::postprocessNAS(const cv::Size &originalSize,
                             const cv::Size &resizedShape,
                             const std::vector<Ort::Value> &outputTensors,
                             float confThreshold, float iouThreshold) {
  std::vector<Detection> detections;

  if (outputTensors.size() < 2) {
    return postprocessStandard(originalSize, resizedShape, outputTensors,
                               confThreshold, iouThreshold);
  }

  const float *boxOutput = outputTensors[0].GetTensorData<float>();
  const float *scoreOutput = outputTensors[1].GetTensorData<float>();
  const std::vector<int64_t> boxShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  const std::vector<int64_t> scoreShape =
      outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();

  const int numDetections = static_cast<int>(boxShape[1]);
  const int numClasses = static_cast<int>(scoreShape[2]);

  // Pre-compute scale and padding
  float scale, padX, padY;
  preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
  const float invScale = 1.0f / scale;

  std::vector<BoundingBox> boxes;
  std::vector<float> confs;
  std::vector<int> classIds;
  boxes.reserve(256);
  confs.reserve(256);
  classIds.reserve(256);

  for (int i = 0; i < numDetections; ++i) {
    // Find max class first (allows early continue if below threshold)
    int classId = 0;
    float maxScore = scoreOutput[i * numClasses];
    for (int c = 1; c < numClasses; ++c) {
      const float score = scoreOutput[i * numClasses + c];
      if (score > maxScore) {
        maxScore = score;
        classId = c;
      }
    }

    if (maxScore <= confThreshold)
      continue;

    const float x1 = (boxOutput[i * 4 + 0] - padX) * invScale;
    const float y1 = (boxOutput[i * 4 + 1] - padY) * invScale;
    const float x2 = (boxOutput[i * 4 + 2] - padX) * invScale;
    const float y2 =
        (boxOutput[i * 4 + 3] - padY) * invScale; // WAIT, NAS specific?

    BoundingBox box;
    box.x = utils::clamp(static_cast<int>(x1), 0, originalSize.width - 1);
    box.y = utils::clamp(static_cast<int>(y1), 0, originalSize.height - 1);
    box.width =
        utils::clamp(static_cast<int>(x2 - x1), 1, originalSize.width - box.x);
    box.height =
        utils::clamp(static_cast<int>(y2 - y1), 1, originalSize.height - box.y);

    boxes.push_back(box);
    confs.push_back(maxScore);
    classIds.push_back(classId);
  }

  std::vector<int> indices;
  nms::NMSBoxesBatched(boxes, confs, classIds, confThreshold, iouThreshold,
                       indices);

  detections.reserve(indices.size());
  for (int idx : indices) {
    detections.emplace_back(boxes[idx], confs[idx], classIds[idx]);
  }

  return detections;
}

} // namespace det
} // namespace yolos
