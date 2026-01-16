#pragma once

// One-file unified YOLO classifier (merges YOLO11 and YOLO12 implementations)

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <string>
#include <variant>
#include <vector>

#include "utils/YOLO-Utils.hpp"

// Using standardized ClassificationResult from YOLO-Utils.hpp

class BaseYOLOClassifier {
public:
  BaseYOLOClassifier(const std::string &modelPath,
                     const std::string &labelsPath, bool useGPU = false,
                     const cv::Size &targetInputShape = cv::Size(224, 224));
  ClassificationResult classify(const cv::Mat &image);
  void drawResult(cv::Mat &image, const ClassificationResult &result,
                  const cv::Point &position = cv::Point(10, 10)) const {
    utils::DrawingUtils::drawClassificationResult(image, result, position);
  }
  cv::Size getInputShape() const { return inputImageShape_; }
  bool isModelInputShapeDynamic() const { return isDynamicInputShape_; }

private:
  Ort::Env env_{nullptr};
  Ort::SessionOptions sessionOptions_{nullptr};
  Ort::Session session_{nullptr};
  bool isDynamicInputShape_{};
  cv::Size inputImageShape_{};
  std::vector<float>
      inputBuffer_{}; // persistent input buffer to avoid reallocations
  std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings_{};
  std::vector<const char *> inputNames_{};
  std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings_{};
  std::vector<const char *> outputNames_{};
  size_t numInputNodes_{}, numOutputNodes_{};
  int numClasses_{0};
  std::vector<std::string> classNames_{};
  void preprocess(const cv::Mat &image, float *&blob,
                  std::vector<int64_t> &inputTensorShape);
  ClassificationResult
  postprocess(const std::vector<Ort::Value> &outputTensors);
};

// Thin wrappers for versioned classifiers
class YOLO11Classifier : public BaseYOLOClassifier {
public:
  using BaseYOLOClassifier::BaseYOLOClassifier;
};

class YOLO12Classifier : public BaseYOLOClassifier {
public:
  using BaseYOLOClassifier::BaseYOLOClassifier;
};

enum class YOLOClassVersion { V11, V12 };

class YOLOClassifier {
public:
  YOLOClassifier(const std::string &modelPath, const std::string &labelsPath,
                 bool useGPU = false,
                 YOLOClassVersion version = YOLOClassVersion::V11) {
    if (version == YOLOClassVersion::V11) {
      impl_.template emplace<YOLO11Classifier>(modelPath, labelsPath, useGPU);
    } else {
      impl_.template emplace<YOLO12Classifier>(modelPath, labelsPath, useGPU);
    }
  }
  ClassificationResult classify(const cv::Mat &image) {
    if (auto *p = std::get_if<YOLO11Classifier>(&impl_))
      return p->classify(image);
    if (auto *q = std::get_if<YOLO12Classifier>(&impl_))
      return q->classify(image);
    return {};
  }
  void drawResult(cv::Mat &image, const ClassificationResult &result,
                  const cv::Point &position = cv::Point(10, 10)) const {
    if (auto *p = std::get_if<YOLO11Classifier>(&impl_)) {
      p->drawResult(image, result, position);
      return;
    }
    if (auto *q = std::get_if<YOLO12Classifier>(&impl_)) {
      q->drawResult(image, result, position);
      return;
    }
  }
  cv::Size getInputShape() const {
    if (auto *p = std::get_if<YOLO11Classifier>(&impl_))
      return p->getInputShape();
    if (auto *q = std::get_if<YOLO12Classifier>(&impl_))
      return q->getInputShape();
    return cv::Size();
  }
  bool isModelInputShapeDynamic() const {
    if (auto *p = std::get_if<YOLO11Classifier>(&impl_))
      return p->isModelInputShapeDynamic();
    if (auto *q = std::get_if<YOLO12Classifier>(&impl_))
      return q->isModelInputShapeDynamic();
    return true;
  }

private:
  std::variant<std::monostate, YOLO11Classifier, YOLO12Classifier> impl_;
};
