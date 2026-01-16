#include "class/YOLOCLASS.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <thread>

// Implementation of BaseYOLOClassifier

BaseYOLOClassifier::BaseYOLOClassifier(const std::string &modelPath,
                                       const std::string &labelsPath,
                                       bool useGPU,
                                       const cv::Size &targetInputShape)
    : inputImageShape_(targetInputShape) {
  env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_CLASSIFICATION_ENV");
  sessionOptions_ = Ort::SessionOptions();
  sessionOptions_.SetIntraOpNumThreads(
      std::min(4, static_cast<int>(std::thread::hardware_concurrency())));
  sessionOptions_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
  auto cudaAvailable =
      std::find(availableProviders.begin(), availableProviders.end(),
                "CUDAExecutionProvider");
  OrtCUDAProviderOptions cudaOption{};
  if (useGPU && cudaAvailable != availableProviders.end()) {
    sessionOptions_.AppendExecutionProvider_CUDA(cudaOption);
  }
#ifdef _WIN32
  std::wstring w_modelPath = std::wstring(modelPath.begin(), modelPath.end());
  session_ = Ort::Session(env_, w_modelPath.c_str(), sessionOptions_);
#else
  session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
#endif
  Ort::AllocatorWithDefaultOptions allocator;
  numInputNodes_ = session_.GetInputCount();
  numOutputNodes_ = session_.GetOutputCount();
  auto input_node_name = session_.GetInputNameAllocated(0, allocator);
  inputNodeNameAllocatedStrings_.push_back(std::move(input_node_name));
  inputNames_.push_back(inputNodeNameAllocatedStrings_.back().get());
  Ort::TypeInfo inputTypeInfo = session_.GetInputTypeInfo(0);
  auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> modelInputTensorShapeVec = inputTensorInfo.GetShape();
  if (modelInputTensorShapeVec.size() == 4) {
    isDynamicInputShape_ = (modelInputTensorShapeVec[2] == -1 ||
                            modelInputTensorShapeVec[3] == -1);
    if (!isDynamicInputShape_) {
      int modelH = static_cast<int>(modelInputTensorShapeVec[2]);
      int modelW = static_cast<int>(modelInputTensorShapeVec[3]);
      if (modelH != inputImageShape_.height ||
          modelW != inputImageShape_.width) {
        std::cout << "Warning: Target preprocessing shape ("
                  << inputImageShape_.height << "x" << inputImageShape_.width
                  << ") differs from model's fixed input shape (" << modelH
                  << "x" << modelW << "). "
                  << "Image will be preprocessed to " << inputImageShape_.height
                  << "x" << inputImageShape_.width << "." << std::endl;
      }
    }
  } else {
    isDynamicInputShape_ = true;
  }
  auto output_node_name = session_.GetOutputNameAllocated(0, allocator);
  outputNodeNameAllocatedStrings_.push_back(std::move(output_node_name));
  outputNames_.push_back(outputNodeNameAllocatedStrings_.back().get());
  Ort::TypeInfo outputTypeInfo = session_.GetOutputTypeInfo(0);
  auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> outputTensorShapeVec = outputTensorInfo.GetShape();
  if (!outputTensorShapeVec.empty()) {
    if (outputTensorShapeVec.size() == 2 && outputTensorShapeVec[0] > 0) {
      numClasses_ = static_cast<int>(outputTensorShapeVec[1]);
    } else if (outputTensorShapeVec.size() == 1 &&
               outputTensorShapeVec[0] > 0) {
      numClasses_ = static_cast<int>(outputTensorShapeVec[0]);
    } else {
      for (long long dim : outputTensorShapeVec)
        if (dim > 1 && numClasses_ == 0)
          numClasses_ = static_cast<int>(dim);
      if (numClasses_ == 0 && !outputTensorShapeVec.empty())
        numClasses_ = static_cast<int>(outputTensorShapeVec.back());
    }
  }
  classNames_ = utils::getClassNames(labelsPath);
}

void BaseYOLOClassifier::preprocess(const cv::Mat &image, float *&blob,
                                    std::vector<int64_t> &inputTensorShape) {
#ifdef TIMING_MODE
  ScopedTimer timer("Preprocessing (Ultralytics-style)");
#endif
  if (image.empty())
    throw std::runtime_error("Input image to preprocess is empty.");

  // Classification preprocessing: resize shortest side to target size, then
  // center crop This matches Ultralytics' classify_transforms behavior
  int target_size = inputImageShape_.width; // Assuming square input (224x224)
  int h = image.rows;
  int w = image.cols;

  // Resize: shortest side to target_size, maintaining aspect ratio
  int new_h, new_w;
  if (h < w) {
    new_h = target_size;
    new_w =
        static_cast<int>(std::round(w * (static_cast<float>(target_size) / h)));
  } else {
    new_w = target_size;
    new_h =
        static_cast<int>(std::round(h * (static_cast<float>(target_size) / w)));
  }

  cv::Mat rgbImage;
  cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

  cv::Mat resized;
  cv::resize(rgbImage, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  // Center crop to target_size x target_size
  int y_start = std::max(0, (new_h - target_size) / 2);
  int x_start = std::max(0, (new_w - target_size) / 2);
  cv::Mat cropped =
      resized(cv::Rect(x_start, y_start, target_size, target_size));

  // Convert BGR to RGB
  cv::Mat rgbImageMat;
  cv::cvtColor(cropped, rgbImageMat, cv::COLOR_BGR2RGB);

  // Normalize to [0, 1]
  cv::Mat floatRgbImage;
  rgbImageMat.convertTo(floatRgbImage, CV_32F, 1.0 / 255.0);

  inputTensorShape = {1, 3, static_cast<int64_t>(floatRgbImage.rows),
                      static_cast<int64_t>(floatRgbImage.cols)};
  const int final_h = static_cast<int>(inputTensorShape[2]);
  const int final_w = static_cast<int>(inputTensorShape[3]);
  const size_t tensorSize = static_cast<size_t>(1) * 3 * final_h * final_w;
  inputBuffer_.resize(tensorSize);

  // Convert HWC to CHW format
  std::vector<cv::Mat> channels(3);
  cv::split(floatRgbImage, channels);
  for (int c = 0; c < 3; ++c) {
    const cv::Mat &plane = channels[c];
    std::memcpy(inputBuffer_.data() + c * (final_h * final_w),
                plane.ptr<float>(),
                static_cast<size_t>(final_h * final_w) * sizeof(float));
  }
  blob = inputBuffer_.data();
}

ClassificationResult
BaseYOLOClassifier::postprocess(const std::vector<Ort::Value> &outputTensors) {
#ifdef TIMING_MODE
  ScopedTimer timer("Postprocessing");
#endif
  if (outputTensors.empty())
    return {};
  const float *rawOutput = outputTensors[0].GetTensorData<float>();
  if (!rawOutput)
    return {};
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  size_t numScores = utils::MathUtils::vectorProduct(outputShape);
  int currentNumClasses =
      numClasses_ > 0 ? numClasses_ : static_cast<int>(classNames_.size());
  if (currentNumClasses <= 0)
    return {};

  // Note: YOLO classification models exported to ONNX already include Softmax
  // as the last layer, so rawOutput contains probabilities (not logits). No
  // need to apply softmax again.
  int bestClassId = -1;
  float maxProb = -std::numeric_limits<float>::infinity();

  if (outputShape.size() == 2 && outputShape[0] == 1) {
    // Output shape is [1, num_classes]
    for (int i = 0;
         i < currentNumClasses && i < static_cast<int>(outputShape[1]); ++i) {
      if (rawOutput[i] > maxProb) {
        maxProb = rawOutput[i];
        bestClassId = i;
      }
    }
  } else {
    // Output shape is [num_classes]
    for (int i = 0; i < currentNumClasses && i < static_cast<int>(numScores);
         ++i) {
      if (rawOutput[i] > maxProb) {
        maxProb = rawOutput[i];
        bestClassId = i;
      }
    }
  }

  if (bestClassId == -1)
    return {};
  float confidence = maxProb;
  std::string className =
      (bestClassId >= 0 &&
       static_cast<size_t>(bestClassId) < classNames_.size())
          ? classNames_[bestClassId]
          : ("ClassID_" + std::to_string(bestClassId));
  return ClassificationResult(bestClassId, confidence, className);
}

ClassificationResult BaseYOLOClassifier::classify(const cv::Mat &image) {
#ifdef TIMING_MODE
  ScopedTimer timer("Overall classification task");
#endif
  if (image.empty())
    return {};
  float *blobPtr = nullptr;
  std::vector<int64_t> currentInputTensorShape;
  preprocess(image, blobPtr, currentInputTensorShape);
  size_t inputTensorSize =
      utils::MathUtils::vectorProduct(currentInputTensorShape);
  Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, blobPtr, inputTensorSize, currentInputTensorShape.data(),
      currentInputTensorShape.size());
  std::vector<Ort::Value> outputTensors =
      session_.Run(Ort::RunOptions{nullptr}, inputNames_.data(), &inputTensor,
                   numInputNodes_, outputNames_.data(), numOutputNodes_);
  if (outputTensors.empty())
    return {};
  return postprocess(outputTensors);
}
