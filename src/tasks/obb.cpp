#include "yolos/tasks/obb.hpp"
#include "yolos/core/drawing.hpp"
#include "yolos/core/nms.hpp"
#include "yolos/core/utils.hpp"
#include <iomanip>
#include <sstream>

namespace yolos {
namespace obb {

YOLOOBBDetector::YOLOOBBDetector(const std::string &modelPath,
                                 const std::string &labelsPath, bool useGPU)
    : OrtSessionBase(modelPath, useGPU) {
  classNames_ = utils::getClassNames(labelsPath);
  classColors_ = drawing::generateColors(classNames_);

  // Pre-allocate inference buffer
  buffer_.ensureCapacity(inputShape_.height, inputShape_.width, 3);
}

std::vector<OBBResult> YOLOOBBDetector::detect(const cv::Mat &image,
                                               float confThreshold,
                                               float iouThreshold, int maxDet) {
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
                     iouThreshold, maxDet);
}

void YOLOOBBDetector::drawDetections(cv::Mat &image,
                                     const std::vector<OBBResult> &results,
                                     int thickness) const {
  for (const auto &det : results) {
    if (det.classId >= 0 &&
        static_cast<size_t>(det.classId) < classNames_.size()) {
      std::string label = classNames_[det.classId] + ": " +
                          std::to_string(static_cast<int>(det.conf * 100)) +
                          "%";
      const cv::Scalar &color = classColors_[det.classId % classColors_.size()];
      drawing::drawOrientedBoundingBox(image, det.box, label, color, thickness);
    }
  }
}

std::vector<OBBResult> YOLOOBBDetector::postprocess(
    const cv::Size &originalSize, const cv::Size &resizedShape,
    const std::vector<Ort::Value> &outputTensors, float confThreshold,
    float iouThreshold, int maxDet) {
  const float *rawOutput = outputTensors[0].GetTensorData<float>();
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

  // Detect output format based on shape:
  // YOLOv8/v11: [1, num_features, num_detections] - requires NMS
  // YOLO26:     [1, 300, 7] - end-to-end, NMS-free
  if (outputShape.size() == 3 && outputShape[2] == 7) {
    // YOLO26 end-to-end format: [1, num_detections, 7]
    return postprocessV26(originalSize, resizedShape, rawOutput, outputShape,
                          confThreshold, maxDet);
  } else {
    // YOLOv8/v11 format: [1, num_features, num_detections]
    return postprocessV8(originalSize, resizedShape, rawOutput, outputShape,
                         confThreshold, iouThreshold, maxDet);
  }
}

std::vector<OBBResult> YOLOOBBDetector::postprocessV8(
    const cv::Size &originalSize, const cv::Size &resizedShape,
    const float *rawOutput, const std::vector<int64_t> &outputShape,
    float confThreshold, float iouThreshold, int maxDet) {
  std::vector<OBBResult> results;

  const int numFeatures = static_cast<int>(outputShape[1]);
  const int numDetections = static_cast<int>(outputShape[2]);

  if (numDetections == 0)
    return results;

  // Layout: [x, y, w, h, scores..., angle]
  const int numLabels = numFeatures - 5;
  if (numLabels <= 0)
    return results;

  // Pre-compute letterbox parameters for descaling AFTER NMS
  float scale, padX, padY;
  preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
  const float invScale = 1.0f / scale;

  // Transpose output for easier access
  cv::Mat output = cv::Mat(numFeatures, numDetections, CV_32F,
                           const_cast<float *>(rawOutput));
  output = output.t();

  // Collect boxes in LETTERBOX coordinates (NMS applied before descaling)
  std::vector<OrientedBoundingBox> letterboxBoxes;
  std::vector<float> confidences;
  std::vector<int> classIds;
  letterboxBoxes.reserve(256);
  confidences.reserve(256);
  classIds.reserve(256);

  for (int i = 0; i < numDetections; ++i) {
    const float *row = output.ptr<float>(i);

    // Find max class score
    float maxScore = row[4];
    int classId = 0;
    for (int j = 1; j < numLabels; ++j) {
      const float score = row[4 + j];
      if (score > maxScore) {
        maxScore = score;
        classId = j;
      }
    }

    if (maxScore <= confThreshold)
      continue;

    const float x = row[0];
    const float y = row[1];
    const float w = row[2];
    const float h = row[3];
    const float angle = row[4 + numLabels];

    // Store in LETTERBOX coordinates for NMS
    letterboxBoxes.emplace_back(x, y, w, h, angle);
    confidences.push_back(maxScore);
    classIds.push_back(classId);
  }

  if (letterboxBoxes.empty())
    return results;

  // Apply class-aware rotated NMS on LETTERBOX coordinates
  std::vector<int> keepIndices = nms::NMSRotatedBatched(
      letterboxBoxes, confidences, classIds, iouThreshold, maxDet);

  results.reserve(keepIndices.size());
  for (int idx : keepIndices) {
    // NOW descale box coordinates from letterbox to original
    const OrientedBoundingBox &lbBox = letterboxBoxes[idx];
    const float cx = (lbBox.x - padX) * invScale;
    const float cy = (lbBox.y - padY) * invScale;
    const float bw = lbBox.width * invScale;
    const float bh = lbBox.height * invScale;

    results.emplace_back(OrientedBoundingBox(cx, cy, bw, bh, lbBox.angle),
                         confidences[idx], classIds[idx]);
  }

  return results;
}

std::vector<OBBResult> YOLOOBBDetector::postprocessV26(
    const cv::Size &originalSize, const cv::Size &resizedShape,
    const float *rawOutput, const std::vector<int64_t> &outputShape,
    float confThreshold, int maxDet) {
  std::vector<OBBResult> results;

  const size_t numDetections = outputShape[1]; // 300
  const size_t numFeatures = outputShape[2];   // 7

  // Pre-compute letterbox parameters
  float scale, padX, padY;
  preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
  const float invScale = 1.0f / scale;

  for (size_t d = 0;
       d < numDetections && static_cast<int>(results.size()) < maxDet; ++d) {
    const size_t base = d * numFeatures;

    // YOLO26 OBB format: [x, y, w, h, conf, class_id, angle]
    const float x = rawOutput[base + 0];                       // center x
    const float y = rawOutput[base + 1];                       // center y
    const float w = rawOutput[base + 2];                       // width
    const float h = rawOutput[base + 3];                       // height
    const float conf = rawOutput[base + 4];                    // confidence
    const int classId = static_cast<int>(rawOutput[base + 5]); // class id
    const float angle = rawOutput[base + 6]; // angle in radians

    if (conf < confThreshold)
      continue;

    // Convert to original image coordinates
    const float cx = (x - padX) * invScale;
    const float cy = (y - padY) * invScale;
    const float bw = w * invScale;
    const float bh = h * invScale;

    results.emplace_back(OrientedBoundingBox(cx, cy, bw, bh, angle), conf,
                         classId);
  }

  return results;
}

} // namespace obb
} // namespace yolos
