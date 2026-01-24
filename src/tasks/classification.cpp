#include "yolos/tasks/classification.hpp"
#include "yolos/core/types.hpp"
#include "yolos/core/utils.hpp"
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace yolos {
namespace cls {

YOLOClassifier::YOLOClassifier(const std::string &modelPath,
                               const std::string &labelsPath, bool useGPU)
    : OrtSessionBase(modelPath, useGPU) {
  classNames_ = utils::getClassNames(labelsPath);

  // Pre-allocate inference buffer
  buffer_.ensureCapacity(inputShape_.height, inputShape_.width, 3);
}

ClassificationResult YOLOClassifier::classify(const cv::Mat &image) {
  if (image.empty())
    return {};

  // Optimized preprocessing with buffer reuse
  cv::Size actualSize;
  preprocessing::letterBoxToBlob(image, buffer_, inputShape_, actualSize,
                                 isDynamicInputShape());

  // Create input tensor
  std::vector<int64_t> inputTensorShape = {1, 3, actualSize.height,
                                           actualSize.width};
  Ort::Value inputTensor =
      createInputTensor(buffer_.blob.data(), inputTensorShape);

  // Run inference
  std::vector<Ort::Value> outputTensors = runInference(inputTensor);

  // Postprocess
  return postprocess(outputTensors);
}

void YOLOClassifier::drawResult(cv::Mat &image,
                                const ClassificationResult &result,
                                const cv::Point &position) const {
  drawClassificationResult(image, result, position);
}

void drawClassificationResult(cv::Mat &image,
                              const ClassificationResult &result,
                              const cv::Point &position,
                              const cv::Scalar &textColor,
                              const cv::Scalar &bgColor) {
  if (image.empty() || result.classId == -1)
    return;

  std::ostringstream ss;
  ss << result.className << ": " << std::fixed << std::setprecision(1)
     << result.confidence * 100 << "%";
  std::string text = ss.str();

  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = std::min(image.rows, image.cols) * 0.001;
  fontScale = std::max(fontScale, 0.5);
  int thickness = std::max(1, static_cast<int>(fontScale * 2));
  int baseline = 0;

  cv::Size textSize =
      cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

  cv::Point textPos = position;
  textPos.y = std::max(textPos.y, textSize.height + 5);

  cv::Point bgTopLeft(textPos.x - 2, textPos.y - textSize.height - 5);
  cv::Point bgBottomRight(textPos.x + textSize.width + 2, textPos.y + 5);

  bgTopLeft.x = utils::clamp(bgTopLeft.x, 0, image.cols - 1);
  bgTopLeft.y = utils::clamp(bgTopLeft.y, 0, image.rows - 1);
  bgBottomRight.x = utils::clamp(bgBottomRight.x, 0, image.cols - 1);
  bgBottomRight.y = utils::clamp(bgBottomRight.y, 0, image.rows - 1);

  cv::rectangle(image, bgTopLeft, bgBottomRight, bgColor, cv::FILLED);
  cv::putText(image, text, textPos, fontFace, fontScale, textColor, thickness,
              cv::LINE_AA);
}

void YOLOClassifier::preprocess(const cv::Mat &image,
                                std::vector<int64_t> &inputTensorShape) {
  // This is now handled by letterBoxToBlob in classify(),
  // but we'll implement it for compatibility or as a placeholder if needed
  (void)image;
  (void)inputTensorShape;
}

ClassificationResult
YOLOClassifier::postprocess(const std::vector<Ort::Value> &outputTensors) {
  const float *rawOutput = outputTensors[0].GetTensorData<float>();
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

  // Shape is usually [1, num_classes] or [num_classes]
  const size_t numClasses =
      outputShape.size() >= 2 ? outputShape[1] : outputShape[0];

  // Find max score
  int bestClassId = 0;
  float maxProb = rawOutput[0];

  for (size_t i = 1; i < numClasses; ++i) {
    if (rawOutput[i] > maxProb) {
      maxProb = rawOutput[i];
      bestClassId = static_cast<int>(i);
    }
  }

  std::string name = (bestClassId >= 0 &&
                      static_cast<size_t>(bestClassId) < classNames_.size())
                         ? classNames_[bestClassId]
                         : ("Class_" + std::to_string(bestClassId));

  return ClassificationResult(bestClassId, maxProb, name);
}

} // namespace cls
} // namespace yolos
