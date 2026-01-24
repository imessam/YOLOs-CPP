#include "yolos/tasks/segmentation.hpp"
#include "yolos/core/drawing.hpp"
#include "yolos/core/nms.hpp"
#include "yolos/core/utils.hpp"
#include <algorithm>
#include <cmath>

namespace yolos {
namespace seg {

YOLOSegDetector::YOLOSegDetector(const std::string &modelPath,
                                 const std::string &labelsPath, bool useGPU)
    : OrtSessionBase(modelPath, useGPU) {

  // Validate output count for segmentation models
  if (numOutputNodes_ != 2) {
    throw std::runtime_error(
        "Expected 2 output nodes for segmentation model (output0 and output1)");
  }

  classNames_ = utils::getClassNames(labelsPath);
  classColors_ = drawing::generateColors(classNames_);

  // Pre-allocate inference buffer
  buffer_.ensureCapacity(inputShape_.height, inputShape_.width, 3);
}

std::vector<Segmentation> YOLOSegDetector::segment(const cv::Mat &image,
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

void YOLOSegDetector::drawSegmentations(
    cv::Mat &image, const std::vector<Segmentation> &results,
    float maskAlpha) const {
  for (const auto &seg : results) {
    if (seg.classId < 0 ||
        static_cast<size_t>(seg.classId) >= classNames_.size()) {
      continue;
    }

    const cv::Scalar &color = classColors_[seg.classId % classColors_.size()];

    // Draw mask
    if (!seg.mask.empty()) {
      drawing::drawSegmentationMask(image, seg.mask, color, maskAlpha);
    }

    // Draw bounding box and label
    std::string label = classNames_[seg.classId] + ": " +
                        std::to_string(static_cast<int>(seg.conf * 100)) + "%";
    drawing::drawBoundingBox(image, seg.box, label, color);
  }
}

void YOLOSegDetector::drawMasksOnly(cv::Mat &image,
                                    const std::vector<Segmentation> &results,
                                    float maskAlpha) const {
  for (const auto &seg : results) {
    if (seg.classId < 0 ||
        static_cast<size_t>(seg.classId) >= classNames_.size()) {
      continue;
    }
    const cv::Scalar &color = classColors_[seg.classId % classColors_.size()];
    if (!seg.mask.empty()) {
      drawing::drawSegmentationMask(image, seg.mask, color, maskAlpha);
    }
  }
}

std::vector<Segmentation>
YOLOSegDetector::postprocess(const cv::Size &originalSize,
                             const cv::Size &letterboxSize,
                             const std::vector<Ort::Value> &outputTensors,
                             float confThreshold, float iouThreshold) {
  std::vector<Segmentation> results;

  if (outputTensors.size() < 2) {
    return results;
  }

  const float *output0 = outputTensors[0].GetTensorData<float>();
  const float *output1 = outputTensors[1].GetTensorData<float>();

  auto shape0 = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  auto shape1 = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();

  if (shape1.size() != 4 || shape1[1] != 32) {
    throw std::runtime_error(
        "Unexpected mask output shape. Expected [1, 32, maskH, maskW]");
  }

  const bool isV26Format = (shape0.size() == 3 && shape0[2] == 38);

  if (isV26Format) {
    return postprocessV26(originalSize, letterboxSize, output0, output1, shape0,
                          shape1, confThreshold);
  }

  const size_t numFeatures = shape0[1];
  const size_t numDetections = shape0[2];

  if (numDetections == 0)
    return results;

  const int numClasses = static_cast<int>(numFeatures) - 4 - 32;
  if (numClasses <= 0)
    return results;

  const int maskH = static_cast<int>(shape1[2]);
  const int maskW = static_cast<int>(shape1[3]);

  std::vector<cv::Mat> prototypeMasks;
  prototypeMasks.reserve(32);
  for (int m = 0; m < 32; ++m) {
    cv::Mat proto(maskH, maskW, CV_32F,
                  const_cast<float *>(output1 + m * maskH * maskW));
    prototypeMasks.emplace_back(proto.clone());
  }

  float gain, padW, padH;
  preprocessing::getScalePad(originalSize, letterboxSize, gain, padW, padH);
  const float invGain = 1.0f / gain;

  std::vector<cv::Rect2f> letterboxBoxes;
  std::vector<float> confidences;
  std::vector<int> classIds;
  std::vector<std::vector<float>> maskCoeffsList;
  letterboxBoxes.reserve(256);
  confidences.reserve(256);
  classIds.reserve(256);
  maskCoeffsList.reserve(256);

  for (size_t i = 0; i < numDetections; ++i) {
    const float xc = output0[0 * numDetections + i];
    const float yc = output0[1 * numDetections + i];
    const float w = output0[2 * numDetections + i];
    const float h = output0[3 * numDetections + i];

    int classId = 0;
    float maxConf = output0[4 * numDetections + i];
    for (int c = 1; c < numClasses; ++c) {
      const float conf = output0[(4 + c) * numDetections + i];
      if (conf > maxConf) {
        maxConf = conf;
        classId = c;
      }
    }

    if (maxConf < confThreshold)
      continue;

    letterboxBoxes.push_back(cv::Rect2f(xc - w * 0.5f, yc - h * 0.5f, w, h));
    confidences.push_back(maxConf);
    classIds.push_back(classId);

    std::vector<float> maskCoeffs(32);
    for (int m = 0; m < 32; ++m) {
      maskCoeffs[m] = output0[(4 + numClasses + m) * numDetections + i];
    }
    maskCoeffsList.emplace_back(std::move(maskCoeffs));
  }

  if (letterboxBoxes.empty())
    return results;

  std::vector<int> nmsIndices;
  nms::NMSBoxesFBatched(letterboxBoxes, confidences, classIds, confThreshold,
                        iouThreshold, nmsIndices);

  float maskScaleX = static_cast<float>(maskW) / letterboxSize.width;
  float maskScaleY = static_cast<float>(maskH) / letterboxSize.height;

  results.reserve(nmsIndices.size());
  for (int idx : nmsIndices) {
    Segmentation seg;
    const cv::Rect2f &lbBox = letterboxBoxes[idx];
    const float left = (lbBox.x - padW) * invGain;
    const float top = (lbBox.y - padH) * invGain;
    const float scaledW = lbBox.width * invGain;
    const float scaledH = lbBox.height * invGain;

    seg.box.x = utils::clamp(static_cast<int>(left), 0, originalSize.width - 1);
    seg.box.y = utils::clamp(static_cast<int>(top), 0, originalSize.height - 1);
    seg.box.width = utils::clamp(static_cast<int>(scaledW), 1,
                                 originalSize.width - seg.box.x);
    seg.box.height = utils::clamp(static_cast<int>(scaledH), 1,
                                  originalSize.height - seg.box.y);
    seg.conf = confidences[idx];
    seg.classId = classIds[idx];

    cv::Mat finalMask = cv::Mat::zeros(maskH, maskW, CV_32F);
    for (int m = 0; m < 32; ++m) {
      finalMask += maskCoeffsList[idx][m] * prototypeMasks[m];
    }

    cv::exp(-finalMask, finalMask);
    finalMask = 1.0 / (1.0 + finalMask);

    int x1 = std::max(
        0, std::min(static_cast<int>(std::round((padW - 0.1f) * maskScaleX)),
                    maskW - 1));
    int y1 = std::max(
        0, std::min(static_cast<int>(std::round((padH - 0.1f) * maskScaleY)),
                    maskH - 1));
    int x2 = std::max(
        x1, std::min(static_cast<int>(std::round(
                         (letterboxSize.width - padW + 0.1f) * maskScaleX)),
                     maskW));
    int y2 = std::max(
        y1, std::min(static_cast<int>(std::round(
                         (letterboxSize.height - padH + 0.1f) * maskScaleY)),
                     maskH));

    if (x2 <= x1 || y2 <= y1)
      continue;

    cv::Mat croppedMask = finalMask(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    cv::Mat resizedMask;
    cv::resize(croppedMask, resizedMask, originalSize, 0, 0, cv::INTER_LINEAR);

    cv::Mat binaryMask;
    cv::threshold(resizedMask, binaryMask, MASK_THRESHOLD, 255.0,
                  cv::THRESH_BINARY);
    binaryMask.convertTo(binaryMask, CV_8U);

    cv::Mat finalBinaryMask = cv::Mat::zeros(originalSize, CV_8U);
    cv::Rect roi(seg.box.x, seg.box.y, seg.box.width, seg.box.height);
    roi &= cv::Rect(0, 0, binaryMask.cols, binaryMask.rows);
    if (roi.area() > 0) {
      binaryMask(roi).copyTo(finalBinaryMask(roi));
    }

    seg.mask = finalBinaryMask;
    results.push_back(seg);
  }

  return results;
}

std::vector<Segmentation> YOLOSegDetector::postprocessV26(
    const cv::Size &originalSize, const cv::Size &letterboxSize,
    const float *output0, const float *output1,
    const std::vector<int64_t> &shape0, const std::vector<int64_t> &shape1,
    float confThreshold) {
  std::vector<Segmentation> results;
  const size_t numDetections = shape0[1];
  const size_t numFeaturesPerDet = shape0[2];

  if (numDetections == 0 || numFeaturesPerDet != 38)
    return results;

  const int maskH = static_cast<int>(shape1[2]);
  const int maskW = static_cast<int>(shape1[3]);

  std::vector<cv::Mat> prototypeMasks;
  prototypeMasks.reserve(32);
  for (int m = 0; m < 32; ++m) {
    cv::Mat proto(maskH, maskW, CV_32F,
                  const_cast<float *>(output1 + m * maskH * maskW));
    prototypeMasks.emplace_back(proto.clone());
  }

  float gain, padW, padH;
  preprocessing::getScalePad(originalSize, letterboxSize, gain, padW, padH);
  const float invGain = 1.0f / gain;

  float maskScaleX = static_cast<float>(maskW) / letterboxSize.width;
  float maskScaleY = static_cast<float>(maskH) / letterboxSize.height;

  for (size_t i = 0; i < numDetections; ++i) {
    const float *det = output0 + i * numFeaturesPerDet;
    const float conf = det[4];
    if (conf < confThreshold)
      continue;

    const float x1_lb = det[0];
    const float y1_lb = det[1];
    const float x2_lb = det[2];
    const float y2_lb = det[3];
    const int classId = static_cast<int>(det[5]);

    Segmentation seg;
    seg.box.x = utils::clamp(static_cast<int>((x1_lb - padW) * invGain), 0,
                             originalSize.width - 1);
    seg.box.y = utils::clamp(static_cast<int>((y1_lb - padH) * invGain), 0,
                             originalSize.height - 1);
    const int x2_scaled = utils::clamp(
        static_cast<int>((x2_lb - padW) * invGain), 0, originalSize.width - 1);
    const int y2_scaled = utils::clamp(
        static_cast<int>((y2_lb - padH) * invGain), 0, originalSize.height - 1);
    seg.box.width = std::max(1, x2_scaled - seg.box.x);
    seg.box.height = std::max(1, y2_scaled - seg.box.y);
    seg.conf = conf;
    seg.classId = classId;

    cv::Mat finalMask = cv::Mat::zeros(maskH, maskW, CV_32F);
    for (int m = 0; m < 32; ++m) {
      finalMask += det[6 + m] * prototypeMasks[m];
    }

    cv::exp(-finalMask, finalMask);
    finalMask = 1.0 / (1.0 + finalMask);

    int mx1 = std::max(
        0, std::min(static_cast<int>(std::round((padW - 0.1f) * maskScaleX)),
                    maskW - 1));
    int my1 = std::max(
        0, std::min(static_cast<int>(std::round((padH - 0.1f) * maskScaleY)),
                    maskH - 1));
    int mx2 = std::max(
        mx1, std::min(static_cast<int>(std::round(
                          (letterboxSize.width - padW + 0.1f) * maskScaleX)),
                      maskW));
    int my2 = std::max(
        my1, std::min(static_cast<int>(std::round(
                          (letterboxSize.height - padH + 0.1f) * maskScaleY)),
                      maskH));

    if (mx2 <= mx1 || my2 <= my1)
      continue;

    cv::Mat croppedMask =
        finalMask(cv::Rect(mx1, my1, mx2 - mx1, my2 - my1)).clone();
    cv::Mat resizedMask;
    cv::resize(croppedMask, resizedMask, originalSize, 0, 0, cv::INTER_LINEAR);

    cv::Mat binaryMask;
    cv::threshold(resizedMask, binaryMask, MASK_THRESHOLD, 255.0,
                  cv::THRESH_BINARY);
    binaryMask.convertTo(binaryMask, CV_8U);

    cv::Mat finalBinaryMask = cv::Mat::zeros(originalSize, CV_8U);
    cv::Rect roi(seg.box.x, seg.box.y, seg.box.width, seg.box.height);
    roi &= cv::Rect(0, 0, binaryMask.cols, binaryMask.rows);
    if (roi.area() > 0) {
      binaryMask(roi).copyTo(finalBinaryMask(roi));
    }

    seg.mask = finalBinaryMask;
    results.push_back(seg);
  }

  return results;
}

} // namespace seg
} // namespace yolos
