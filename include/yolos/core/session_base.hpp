#pragma once

// ============================================================================
// YOLO ONNX Session Base
// ============================================================================
// Common ONNX Runtime session setup and management for all YOLO detectors.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace yolos {

// ============================================================================
// OrtSessionBase - Common ONNX Runtime session management
// ============================================================================

/// @brief Base class for ONNX Runtime session management
/// Handles model loading, session configuration, and common inference setup
class OrtSessionBase {
public:
  /// @brief Constructor - loads and initializes the ONNX model
  /// @param modelPath Path to the ONNX model file
  /// @param useGPU Whether to use GPU (CUDA) for inference
  /// @param numThreads Number of intra-op threads (0 = auto)
  OrtSessionBase(const std::string &modelPath, bool useGPU = false,
                 int numThreads = 0);

  virtual ~OrtSessionBase() = default;

  // Prevent copying
  OrtSessionBase(const OrtSessionBase &) = delete;
  OrtSessionBase &operator=(const OrtSessionBase &) = delete;

  // Allow moving
  OrtSessionBase(OrtSessionBase &&) = default;
  OrtSessionBase &operator=(OrtSessionBase &&) = default;

  /// @brief Get the input image shape expected by the model
  [[nodiscard]] cv::Size getInputShape() const noexcept { return inputShape_; }

  /// @brief Check if input shape is dynamic
  [[nodiscard]] bool isDynamicInputShape() const noexcept {
    return isDynamicInputShape_;
  }

  /// @brief Check if batch size is dynamic
  [[nodiscard]] bool isDynamicBatchSize() const noexcept {
    return isDynamicBatchSize_;
  }

  /// @brief Get the device being used for inference
  [[nodiscard]] const std::string &getDevice() const noexcept {
    return device_;
  }

  /// @brief Get the number of input nodes
  [[nodiscard]] size_t getNumInputNodes() const noexcept {
    return numInputNodes_;
  }

  /// @brief Get the number of output nodes
  [[nodiscard]] size_t getNumOutputNodes() const noexcept {
    return numOutputNodes_;
  }

protected:
  Ort::Env env_{nullptr};
  Ort::SessionOptions sessionOptions_{nullptr};
  Ort::Session session_{nullptr};

  // Input/output node names
  std::vector<Ort::AllocatedStringPtr> inputNameAllocs_;
  std::vector<const char *> inputNames_;
  std::vector<Ort::AllocatedStringPtr> outputNameAllocs_;
  std::vector<const char *> outputNames_;

  size_t numInputNodes_{0};
  size_t numOutputNodes_{0};

  cv::Size inputShape_;
  bool isDynamicInputShape_{false};
  bool isDynamicBatchSize_{false};
  std::string device_{"cpu"};

  /// @brief Run inference with the given input tensor
  /// @param inputTensor Input tensor
  /// @return Vector of output tensors
  std::vector<Ort::Value> runInference(Ort::Value &inputTensor);

  /// @brief Create an input tensor from a blob
  /// @param blob Pointer to the input data
  /// @param inputTensorShape Shape of the input tensor
  /// @return ONNX Runtime input tensor
  Ort::Value createInputTensor(float *blob,
                               const std::vector<int64_t> &inputTensorShape);

private:
  void initSession(const std::string &modelPath, bool useGPU, int numThreads);
};

} // namespace yolos
