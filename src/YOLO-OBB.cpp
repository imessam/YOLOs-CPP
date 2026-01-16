#include "obb/YOLO-OBB.hpp"
#include "tools/Debug.hpp"
#include "tools/ScopedTimer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <thread>

namespace OBB_NMS {

float computeRotatedIoU(const OrientedBoundingBox &box1,
                        const OrientedBoundingBox &box2) {
  // Convert angle from radians to degrees for OpenCV
  cv::RotatedRect rect1(cv::Point2f(box1.x, box1.y),
                        cv::Size2f(box1.width, box1.height),
                        box1.angle * 180.0f / (float)CV_PI);

  cv::RotatedRect rect2(cv::Point2f(box2.x, box2.y),
                        cv::Size2f(box2.width, box2.height),
                        box2.angle * 180.0f / (float)CV_PI);

  // Compute intersection using OpenCV's built-in function
  std::vector<cv::Point2f> intersectionPoints;
  int result =
      cv::rotatedRectangleIntersection(rect1, rect2, intersectionPoints);

  // No intersection
  if (result == cv::INTERSECT_NONE) {
    return 0.0f;
  }

  // Compute intersection area
  float intersectionArea = 0.0f;
  if (intersectionPoints.size() > 2) {
    intersectionArea = (float)cv::contourArea(intersectionPoints);
  }

  // Compute areas of both boxes
  float area1 = box1.width * box1.height;
  float area2 = box2.width * box2.height;

  // Compute union area
  float unionArea = area1 + area2 - intersectionArea;

  // Avoid division by zero
  if (unionArea < 1e-7f) {
    return 0.0f;
  }

  // Return IoU
  return intersectionArea / unionArea;
}

std::vector<int> nmsRotated(const std::vector<OrientedBoundingBox> &boxes,
                            const std::vector<float> &scores,
                            float iou_threshold) {

  if (boxes.empty()) {
    return std::vector<int>();
  }

  // Create indices and sort by score (descending)
  std::vector<int> indices(boxes.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(),
            [&scores](int a, int b) { return scores[a] > scores[b]; });

  // Track which boxes have been suppressed
  std::vector<bool> suppressed(boxes.size(), false);
  std::vector<int> keep;

  // Iterate through sorted boxes
  for (size_t i = 0; i < indices.size(); i++) {
    int idx = indices[i];

    // Skip if already suppressed
    if (suppressed[idx])
      continue;

    // Keep this box
    keep.push_back(idx);

    // Suppress boxes with high IoU to this box
    for (size_t j = i + 1; j < indices.size(); j++) {
      int idx2 = indices[j];

      // Skip if already suppressed
      if (suppressed[idx2])
        continue;

      // Compute IoU
      float iou = computeRotatedIoU(boxes[idx], boxes[idx2]);

      // Suppress if IoU exceeds threshold
      if (iou >= iou_threshold) {
        suppressed[idx2] = true;
      }
    }
  }

  return keep;
}

std::vector<OBB_Detection>
nonMaxSuppression(const std::vector<OBB_Detection> &detections,
                  float iou_threshold, int max_det) {

  if (detections.empty()) {
    return std::vector<OBB_Detection>();
  }

  // Extract boxes and scores
  std::vector<OrientedBoundingBox> boxes;
  std::vector<float> scores;

  for (const auto &det : detections) {
    boxes.push_back(det.box);
    scores.push_back(det.conf);
  }

  // Perform NMS
  std::vector<int> keep_indices = nmsRotated(boxes, scores, iou_threshold);

  // Limit to max_det
  if (keep_indices.size() > static_cast<size_t>(max_det)) {
    keep_indices.resize(max_det);
  }

  // Build result
  std::vector<OBB_Detection> result;
  result.reserve(keep_indices.size());

  for (int idx : keep_indices) {
    result.push_back(detections[idx]);
  }

  return result;
}

} // namespace OBB_NMS

namespace utils {

void drawBoundingBox(cv::Mat &image,
                     const std::vector<OBB_Detection> &detections,
                     const std::vector<std::string> &classNames,
                     const std::vector<cv::Scalar> &classColors) {
  for (const auto &det : detections) {
    const OrientedBoundingBox &obb = det.box;
    cv::Scalar color = classColors[det.classId % classColors.size()];

    // Create rotated rectangle
    cv::RotatedRect rotatedRect(
        cv::Point2f(obb.x, obb.y), cv::Size2f(obb.width, obb.height),
        obb.angle * 180.0f / (float)CV_PI // Convert radians to degrees
    );

    // Get the four vertices of the rotated rectangle
    cv::Point2f vertices[4];
    rotatedRect.points(vertices);

    // Draw the rotated box
    for (int i = 0; i < 4; i++) {
      cv::line(image, vertices[i], vertices[(i + 1) % 4], color, 2,
               cv::LINE_AA);
    }

    // Prepare label with class name and confidence
    std::string label =
        classNames[det.classId] + ": " + std::to_string(det.conf).substr(0, 4);

    // Calculate text size
    int baseline;
    double fontScale = 0.5;
    int thickness = 1;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX,
                                         fontScale, thickness, &baseline);

    // Position label at the top-left corner of the box
    int x = static_cast<int>(obb.x - obb.width / 2);
    int y = static_cast<int>(obb.y - obb.height / 2) - 5;

    // Ensure label stays within image bounds
    x = std::max(0, std::min(x, image.cols - labelSize.width));
    y = std::max(labelSize.height, std::min(y, image.rows - baseline));

    // Draw label background (darker version of box color)
    cv::Scalar labelBgColor = color * 0.6;
    cv::rectangle(image,
                  cv::Rect(x, y - labelSize.height, labelSize.width,
                           labelSize.height + baseline),
                  labelBgColor, cv::FILLED);

    // Draw label text
    cv::putText(image, label, cv::Point(x, y), cv::FONT_HERSHEY_DUPLEX,
                fontScale, cv::Scalar::all(255), thickness, cv::LINE_AA);
  }
}

} // namespace utils

YOLOOBBDetector::YOLOOBBDetector(const std::string &modelPath,
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

  std::cout << "Input tensor shape: [";
  for (size_t i = 0; i < inputTensorShapeVec.size(); i++) {
    std::cout << inputTensorShapeVec[i];
    if (i < inputTensorShapeVec.size() - 1)
      std::cout << ", ";
  }
  std::cout << "]" << std::endl;

  if (inputTensorShapeVec.size() >= 4) {
    int height = (inputTensorShapeVec[2] == -1)
                     ? 640
                     : static_cast<int>(inputTensorShapeVec[2]);
    int width = (inputTensorShapeVec[3] == -1)
                    ? 640
                    : static_cast<int>(inputTensorShapeVec[3]);

    if (height <= 0 || width <= 0) {
      std::cerr << "Invalid dimensions detected: " << width << "x" << height
                << std::endl;
      height = 640;
      width = 640;
    }

    inputImageShape = cv::Size(width, height);
    std::cout << "Using input shape: " << width << "x" << height << std::endl;
  } else {
    throw std::runtime_error("Invalid input tensor shape.");
  }

  numInputNodes = session.GetInputCount();
  numOutputNodes = session.GetOutputCount();

  classNames = utils::getClassNames(labelsPath);
  classColors = utils::DrawingUtils::generateColors(classNames);

  std::cout << "Model loaded successfully with " << numInputNodes
            << " input nodes and " << numOutputNodes << " output nodes."
            << std::endl;
}

cv::Mat YOLOOBBDetector::preprocess(const cv::Mat &image, float *&blob,
                                    std::vector<int64_t> &inputTensorShape) {
  ScopedTimer timer("preprocessing");

  cv::Mat rgbImage;
  cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

  cv::Mat resizedImage;
  utils::ImagePreprocessingUtils::letterBox(
      rgbImage, resizedImage, inputImageShape, cv::Scalar(114, 114, 114), false,
      true, 32);

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

std::vector<OBB_Detection> YOLOOBBDetector::postprocess(
    const cv::Size &originalImageSize, const cv::Size &resizedImageShape,
    const std::vector<Ort::Value> &outputTensors, float confThreshold,
    float iouThreshold, int topk) {
  ScopedTimer timer("postprocessing");
  std::vector<OBB_Detection> detections;

  const float *rawOutput = outputTensors[0].GetTensorData<float>();
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  int num_features = static_cast<int>(outputShape[1]);
  int num_detections = static_cast<int>(outputShape[2]);
  if (num_detections == 0) {
    return detections;
  }

  int num_labels = num_features - 5;
  if (num_labels <= 0) {
    return detections;
  }

  float inp_w = static_cast<float>(resizedImageShape.width);
  float inp_h = static_cast<float>(resizedImageShape.height);
  float orig_w = static_cast<float>(originalImageSize.width);
  float orig_h = static_cast<float>(originalImageSize.height);
  float r = std::min(inp_h / orig_h, inp_w / orig_w);
  int padw = (int)std::round(orig_w * r);
  int padh = (int)std::round(orig_h * r);
  float dw = (inp_w - (float)padw) / 2.0f;
  float dh = (inp_h - (float)padh) / 2.0f;
  float ratio = 1.0f / r;

  cv::Mat output = cv::Mat(num_features, num_detections, CV_32F,
                           const_cast<float *>(rawOutput));
  output = output.t();

  std::vector<OBB_Detection> detectionsForNMS;
  for (int i = 0; i < num_detections; ++i) {
    float *row_ptr = output.ptr<float>(i);

    float x = row_ptr[0];
    float y = row_ptr[1];
    float w = row_ptr[2];
    float h = row_ptr[3];

    float *scores_ptr = row_ptr + 4;
    float maxScore = -FLT_MAX;
    int classId = -1;
    for (int j = 0; j < num_labels; j++) {
      float score = scores_ptr[j];
      if (score > maxScore) {
        maxScore = score;
        classId = j;
      }
    }

    float angle = row_ptr[4 + num_labels];

    if (maxScore > confThreshold) {
      float cx = (x - dw) * ratio;
      float cy = (y - dh) * ratio;
      float bw = w * ratio;
      float bh = h * ratio;

      OrientedBoundingBox obb(cx, cy, bw, bh, angle);
      detectionsForNMS.emplace_back(OBB_Detection{obb, maxScore, classId});
    }
  }

  std::vector<OBB_Detection> post_nms_detections =
      OBB_NMS::nonMaxSuppression(detectionsForNMS, iouThreshold, topk);

  DEBUG_PRINT("Postprocessing completed");
  return post_nms_detections;
}

std::vector<OBB_Detection> YOLOOBBDetector::detect(const cv::Mat &image,
                                                   float confThreshold,
                                                   float iouThreshold) {
  ScopedTimer timer("Overall detection");

  float *blobPtr = nullptr;
  std::vector<int64_t> inputTensorShape = {1, 3, inputImageShape.height,
                                           inputImageShape.width};

  cv::Mat preprocessedImage = preprocess(image, blobPtr, inputTensorShape);

  size_t inputSize = utils::MathUtils::vectorProduct(inputTensorShape);

  std::vector<float> inputTensorValues(blobPtr, blobPtr + inputSize);

  delete[] blobPtr;

  static Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensorValues.data(), inputSize, inputTensorShape.data(),
      inputTensorShape.size());

  std::vector<Ort::Value> outputTensors =
      session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                  numInputNodes, outputNames.data(), numOutputNodes);

  cv::Size resizedImageShape(static_cast<int>(inputTensorShape[3]),
                             static_cast<int>(inputTensorShape[2]));

  std::vector<OBB_Detection> detections =
      postprocess(image.size(), resizedImageShape, outputTensors, confThreshold,
                  iouThreshold, 300);

  return detections;
}
