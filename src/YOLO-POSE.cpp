#include "pose/YOLO-POSE.hpp"
#include "tools/Debug.hpp"
#include "tools/ScopedTimer.hpp"
#include <algorithm>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <thread>

namespace utils {

void drawPoseEstimation(cv::Mat &image,
                        const std::vector<Detection> &detections,
                        float confidenceThreshold, float kptThreshold) {
  // Calculate dynamic sizes based on image dimensions
  const int min_dim = std::min(image.rows, image.cols);
  const float scale_factor = min_dim / 1280.0f; // Reference 1280px size

  // Dynamic sizing parameters
  const int line_thickness = std::max(1, static_cast<int>(2 * scale_factor));
  const int kpt_radius = std::max(2, static_cast<int>(4 * scale_factor));

  // Define the Ultralytics pose palette (BGR format)
  static const std::vector<cv::Scalar> pose_palette = {
      cv::Scalar(0, 128, 255),   // 0
      cv::Scalar(51, 153, 255),  // 1
      cv::Scalar(102, 178, 255), // 2
      cv::Scalar(0, 230, 230),   // 3
      cv::Scalar(255, 153, 255), // 4
      cv::Scalar(255, 204, 153), // 5
      cv::Scalar(255, 102, 255), // 6
      cv::Scalar(255, 51, 255),  // 7
      cv::Scalar(255, 178, 102), // 8
      cv::Scalar(255, 153, 51),  // 9
      cv::Scalar(153, 153, 255), // 10
      cv::Scalar(102, 102, 255), // 11
      cv::Scalar(51, 51, 255),   // 12
      cv::Scalar(153, 255, 153), // 13
      cv::Scalar(102, 255, 102), // 14
      cv::Scalar(51, 255, 51),   // 15
      cv::Scalar(0, 255, 0),     // 16
      cv::Scalar(255, 0, 0),     // 17
      cv::Scalar(0, 0, 255),     // 18
      cv::Scalar(255, 255, 255)  // 19
  };

  // Define per-keypoint color indices
  static const std::vector<int> kpt_color_indices = {
      16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9};
  // Define per-limb color indices
  static const std::vector<int> limb_color_indices = {
      9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16};

  for (const auto &detection : detections) {
    if (detection.conf < confidenceThreshold)
      continue;

    const auto &box = detection.box;
    cv::rectangle(image, cv::Point(box.x, box.y),
                  cv::Point(box.x + box.width, box.y + box.height),
                  cv::Scalar(0, 255, 0), line_thickness);

    const size_t numKpts = detection.keypoints.size();
    std::vector<cv::Point> kpt_points(numKpts, cv::Point(-1, -1));
    std::vector<bool> valid(numKpts, false);

    for (size_t i = 0; i < numKpts; i++) {
      if (detection.keypoints[i].conf >= kptThreshold) {
        int x = static_cast<int>(std::round(detection.keypoints[i].x));
        int y = static_cast<int>(std::round(detection.keypoints[i].y));
        kpt_points[i] = cv::Point(x, y);
        valid[i] = true;
        int color_index =
            (i < kpt_color_indices.size()) ? kpt_color_indices[i] : 0;
        cv::circle(image, cv::Point(x, y), kpt_radius,
                   pose_palette[color_index], -1, cv::LINE_AA);
      }
    }

    for (size_t j = 0; j < POSE_SKELETON.size(); j++) {
      auto [src, dst] = POSE_SKELETON[j];
      if (src < (int)numKpts && dst < (int)numKpts && valid[src] &&
          valid[dst]) {
        int limb_color_index =
            (j < limb_color_indices.size()) ? limb_color_indices[j] : 0;
        cv::line(image, kpt_points[src], kpt_points[dst],
                 pose_palette[limb_color_index], line_thickness, cv::LINE_AA);
      }
    }
  }
}

} // namespace utils

YOLOPOSEDetector::YOLOPOSEDetector(const std::string &modelPath,
                                   const std::string &labelsPath, bool useGPU) {
  env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
  sessionOptions = Ort::SessionOptions();

  sessionOptions.SetIntraOpNumThreads(
      std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
  auto cudaAvailable =
      std::find(availableProviders.begin(), availableProviders.end(),
                "CUDAExecutionProvider");
  OrtCUDAProviderOptions cudaOption;

  if (useGPU && cudaAvailable != availableProviders.end()) {
    std::cout << "Inference device: GPU" << std::endl;
    sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
  } else {
    if (useGPU) {
      std::cout
          << "GPU is not supported by your ONNXRuntime build. Fallback to CPU."
          << std::endl;
    }
    std::cout << "Inference device: CPU" << std::endl;
  }

#ifdef _WIN32
  std::wstring w_modelPath(modelPath.begin(), modelPath.end());
  session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
  session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

  Ort::AllocatorWithDefaultOptions allocator;

  Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
  std::vector<int64_t> inputTensorShapeVec =
      inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
  isDynamicInputShape =
      (inputTensorShapeVec.size() >= 4) &&
      (inputTensorShapeVec[2] == -1 && inputTensorShapeVec[3] == -1);

  auto input_name = session.GetInputNameAllocated(0, allocator);
  inputNodeNameAllocatedStrings.push_back(std::move(input_name));
  inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

  auto output_name = session.GetOutputNameAllocated(0, allocator);
  outputNodeNameAllocatedStrings.push_back(std::move(output_name));
  outputNames.push_back(outputNodeNameAllocatedStrings.back().get());

  if (inputTensorShapeVec.size() >= 4) {
    inputImageShape = cv::Size(static_cast<int>(inputTensorShapeVec[3]),
                               static_cast<int>(inputTensorShapeVec[2]));
  } else {
    throw std::runtime_error("Invalid input tensor shape.");
  }

  numInputNodes = session.GetInputCount();
  numOutputNodes = session.GetOutputCount();

  std::cout << "Model loaded successfully with " << numInputNodes
            << " input nodes and " << numOutputNodes << " output nodes."
            << std::endl;
}

cv::Mat YOLOPOSEDetector::preprocess(const cv::Mat &image, float *&blob,
                                     std::vector<int64_t> &inputTensorShape) {
  ScopedTimer timer("preprocessing");

  cv::Mat rgbImage;
  cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

  cv::Mat resizedImage;
  utils::ImagePreprocessingUtils::letterBox(
      rgbImage, resizedImage, inputImageShape, cv::Scalar(114, 114, 114),
      isDynamicInputShape, false, true, 32);

  inputTensorShape[2] = resizedImage.rows;
  inputTensorShape[3] = resizedImage.cols;

  resizedImage.convertTo(resizedImage, CV_32FC3, 1 / 255.0f);

  blob = new float[resizedImage.cols * resizedImage.rows *
                   resizedImage.channels()];

  std::vector<cv::Mat> chw(resizedImage.channels());
  for (int i = 0; i < resizedImage.channels(); ++i) {
    chw[i] = cv::Mat(resizedImage.rows, resizedImage.cols, CV_32FC1,
                     blob + i * resizedImage.cols * resizedImage.rows);
  }
  cv::split(resizedImage, chw);

  DEBUG_PRINT("Preprocessing completed")

  return resizedImage;
}

std::vector<Detection>
YOLOPOSEDetector::postprocess(const cv::Size &originalImageSize,
                              const cv::Size &resizedImageShape,
                              const std::vector<Ort::Value> &outputTensors,
                              float confThreshold, float iouThreshold) {
  ScopedTimer timer("postprocessing");
  std::vector<Detection> detections;

  const float *rawOutput = outputTensors[0].GetTensorData<float>();
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

  const size_t numFeatures = outputShape[1];
  const size_t numDetections = outputShape[2];
  const int numKeypoints = 17;
  const int featuresPerKeypoint = 3;

  if (numFeatures != 4 + 1 + numKeypoints * featuresPerKeypoint) {
    std::cerr << "Invalid output shape for pose estimation model" << std::endl;
    return detections;
  }

  const float scale = std::min(
      static_cast<float>(resizedImageShape.width) / originalImageSize.width,
      static_cast<float>(resizedImageShape.height) / originalImageSize.height);
  const cv::Size scaledSize((int)(originalImageSize.width * scale),
                            (int)(originalImageSize.height * scale));
  const cv::Point2f padding(
      ((float)resizedImageShape.width - (float)scaledSize.width) / 2.0f,
      ((float)resizedImageShape.height - (float)scaledSize.height) / 2.0f);

  std::vector<BoundingBox> boxes;
  std::vector<float> confidences;
  std::vector<std::vector<KeyPoint>> allKeypoints;

  for (size_t d = 0; d < numDetections; ++d) {
    const float objConfidence = rawOutput[4 * numDetections + d];
    if (objConfidence < confThreshold)
      continue;

    const float cx = rawOutput[0 * numDetections + d];
    const float cy = rawOutput[1 * numDetections + d];
    const float w = rawOutput[2 * numDetections + d];
    const float h = rawOutput[3 * numDetections + d];

    BoundingBox box;
    box.x = static_cast<int>((cx - padding.x - w / 2) / scale);
    box.y = static_cast<int>((cy - padding.y - h / 2) / scale);
    box.width = static_cast<int>(w / scale);
    box.height = static_cast<int>(h / scale);

    box.x =
        utils::MathUtils::clamp(box.x, 0, originalImageSize.width - box.width);
    box.y = utils::MathUtils::clamp(box.y, 0,
                                    originalImageSize.height - box.height);
    box.width =
        utils::MathUtils::clamp(box.width, 0, originalImageSize.width - box.x);
    box.height = utils::MathUtils::clamp(box.height, 0,
                                         originalImageSize.height - box.y);

    std::vector<KeyPoint> keypoints;
    for (int k = 0; k < numKeypoints; ++k) {
      const int offset = 5 + k * featuresPerKeypoint;
      KeyPoint kpt;
      kpt.x = (rawOutput[offset * numDetections + d] - padding.x) / scale;
      kpt.y = (rawOutput[(offset + 1) * numDetections + d] - padding.y) / scale;
      kpt.conf =
          1.0f /
          (1.0f + std::exp(-rawOutput[(offset + 2) * numDetections + d]));

      kpt.x = utils::MathUtils::clamp(
          kpt.x, 0.0f, static_cast<float>(originalImageSize.width - 1));
      kpt.y = utils::MathUtils::clamp(
          kpt.y, 0.0f, static_cast<float>(originalImageSize.height - 1));

      keypoints.emplace_back(kpt);
    }

    boxes.emplace_back(box);
    confidences.emplace_back(objConfidence);
    allKeypoints.emplace_back(keypoints);
  }

  std::vector<int> indices;
  utils::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);

  for (int idx : indices) {
    Detection det;
    det.box.x = boxes[idx].x;
    det.box.y = boxes[idx].y;
    det.box.width = boxes[idx].width;
    det.box.height = boxes[idx].height;
    det.conf = confidences[idx];
    det.classId = 0;
    det.keypoints = allKeypoints[idx];
    detections.emplace_back(det);
  }

  return detections;
}

std::vector<Detection> YOLOPOSEDetector::detect(const cv::Mat &image,
                                                float confThreshold,
                                                float iouThreshold) {
  ScopedTimer timer("Overall detection");

  float *blobPtr = nullptr;
  std::vector<int64_t> inputTensorShape = {1, 3, inputImageShape.height,
                                           inputImageShape.width};

  cv::Mat preprocessedImage = preprocess(image, blobPtr, inputTensorShape);

  size_t inputTensorSize = utils::MathUtils::vectorProduct(inputTensorShape);

  std::vector<float> inputTensorValues(blobPtr, blobPtr + inputTensorSize);

  delete[] blobPtr;

  static Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensorValues.data(), inputTensorSize,
      inputTensorShape.data(), inputTensorShape.size());

  std::vector<Ort::Value> outputTensors =
      session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                  numInputNodes, outputNames.data(), numOutputNodes);

  cv::Size resizedImageShape(static_cast<int>(inputTensorShape[3]),
                             static_cast<int>(inputTensorShape[2]));

  std::vector<Detection> detections =
      postprocess(image.size(), resizedImageShape, outputTensors, confThreshold,
                  iouThreshold);

  return detections;
}
