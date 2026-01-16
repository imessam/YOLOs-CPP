#include "det/YOLO.hpp"
#include <algorithm>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <thread>

// Implementation of YOLODetector constructor
YOLODetector::YOLODetector(const std::string &modelPath,
                           const std::string &labelsPath, bool useGPU) {
  // Initialize ONNX Runtime environment with warning level
  env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
  sessionOptions = Ort::SessionOptions();

  // Set number of intra-op threads for parallelism
  sessionOptions.SetIntraOpNumThreads(
      std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Retrieve available execution providers (e.g., CPU, CUDA)
  std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
  auto cudaAvailable =
      std::find(availableProviders.begin(), availableProviders.end(),
                "CUDAExecutionProvider");
  OrtCUDAProviderOptions cudaOption;

  // Configure session options based on whether GPU is to be used and available
  if (useGPU && cudaAvailable != availableProviders.end()) {
    std::cout << "Inference device: GPU" << std::endl;
    sessionOptions.AppendExecutionProvider_CUDA(
        cudaOption); // Append CUDA execution provider
    device_used = "gpu";
  } else {
    if (useGPU) {
      std::cout
          << "GPU is not supported by your ONNXRuntime build. Fallback to CPU."
          << std::endl;
    }
    std::cout << "Inference device: CPU" << std::endl;
    device_used = "cpu";
  }

  // Load the ONNX model into the session
#ifdef _WIN32
  std::wstring w_modelPath(modelPath.begin(), modelPath.end());
  session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
  session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

  Ort::AllocatorWithDefaultOptions allocator;

  // Retrieve input tensor shape information
  Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
  std::vector<int64_t> inputTensorShapeVec =
      inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
  isDynamicInputShape =
      (inputTensorShapeVec.size() >= 4) &&
      (inputTensorShapeVec[2] == -1 ||
       inputTensorShapeVec[3] == -1); // Check for dynamic height/width
  isDynamicBatchSize =
      (inputTensorShapeVec.size() >= 4) &&
      (inputTensorShapeVec[0] == -1); // Check for dynamic batch size

  // Allocate and store input node names
  auto input_name = session.GetInputNameAllocated(0, allocator);
  inputNodeNameAllocatedStrings.push_back(std::move(input_name));
  inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

  // Allocate and store output node names
  for (int layer = 0; layer < this->session.GetOutputCount(); layer += 1) {
#if ORT_API_VERSION < 13
    outputNames.push_back(this->session.GetOutputName(layer, allocator));
#else
    Ort::AllocatedStringPtr output_name_Ptr =
        this->session.GetOutputNameAllocated(layer, allocator);
    outputNodeNameAllocatedStrings.push_back(std::move(output_name_Ptr));
    outputNames.push_back(outputNodeNameAllocatedStrings.back().get());
#endif
  }

  // Set the expected input image shape based on the model's input tensor
  if (inputTensorShapeVec.size() >= 4) {
    inputImageShape = cv::Size(static_cast<int>(inputTensorShapeVec[3]),
                               static_cast<int>(inputTensorShapeVec[2]));
    int height = (inputTensorShapeVec[2] == -1)
                     ? 640
                     : static_cast<int>(inputTensorShapeVec[2]);
    int width = (inputTensorShapeVec[3] == -1)
                    ? 640
                    : static_cast<int>(inputTensorShapeVec[3]);
    inputImageShape = cv::Size(width, height);
  } else {
    throw std::runtime_error("Invalid input tensor shape.");
  }

  // Get the number of input and output nodes
  numInputNodes = session.GetInputCount();
  numOutputNodes = session.GetOutputCount();

  // Load class names and generate corresponding colors
  classNames = utils::getClassNames(labelsPath);
  classColors = utils::DrawingUtils::generateColors(classNames);

  std::cout << "Model loaded successfully with " << numInputNodes
            << " input nodes and " << numOutputNodes << " output nodes."
            << std::endl;
}

// Preprocess function implementation
cv::Mat YOLODetector::preprocess(const cv::Mat &image, float *&blob,
                                 std::vector<int64_t> &inputTensorShape) {
#ifdef TIMING_MODE
  ScopedTimer timer("preprocessing");
#endif

  cv::Mat rgbImage;
  cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

  cv::Mat resizedImage;
  // Resize and pad the image using letterBox utility
  utils::ImagePreprocessingUtils::letterBox(
      rgbImage, resizedImage, inputImageShape, cv::Scalar(114, 114, 114),
      isDynamicInputShape, false, true, 32);

  // Update input tensor shape based on resized image dimensions
  inputTensorShape[2] = resizedImage.rows;
  inputTensorShape[3] = resizedImage.cols;

  // Convert image to float and normalize to [0, 1]
  resizedImage.convertTo(resizedImage, CV_32FC3, 1 / 255.0f);

  // Allocate memory for the image blob in CHW format
  blob = new float[resizedImage.cols * resizedImage.rows *
                   resizedImage.channels()];

  // Split the image into separate channels and store in the blob
  std::vector<cv::Mat> chw(resizedImage.channels());
  for (int i = 0; i < resizedImage.channels(); ++i) {
    chw[i] = cv::Mat(resizedImage.rows, resizedImage.cols, CV_32FC1,
                     blob + i * resizedImage.cols * resizedImage.rows);
  }
  cv::split(resizedImage, chw); // Split channels into the blob
#ifdef DEBUG_MODE
  DEBUG_PRINT("Preprocessing completed")
#endif
  return resizedImage;
}

// Batch preprocess function implementation
std::vector<cv::Size>
YOLODetector::batch_preprocess(const std::vector<cv::Mat> &images, float *&blob,
                               std::vector<int64_t> &inputTensorShape) {
#ifdef TIMING_MODE
  ScopedTimer timer("batch preprocessing");
#endif
  size_t batchSize = images.size();
  if (batchSize == 0) {
    blob = nullptr;
    return {};
  }

  // Resize all images to the same size (inputImageShape) with letterbox
  std::vector<cv::Mat> resizedImages(batchSize);
  std::vector<cv::Size> resizedShapes(batchSize);
  for (size_t i = 0; i < batchSize; ++i) {
    cv::Mat rgbImage;
    cv::cvtColor(images[i], rgbImage, cv::COLOR_BGR2RGB);
    utils::ImagePreprocessingUtils::letterBox(
        rgbImage, resizedImages[i], inputImageShape, cv::Scalar(114, 114, 114),
        isDynamicInputShape, false, true, 32);
    resizedShapes[i] = resizedImages[i].size();
  }

  // Update input tensor shape for batch size and image dimensions
  inputTensorShape[0] = static_cast<int64_t>(batchSize);
  inputTensorShape[2] = resizedImages[0].rows;
  inputTensorShape[3] = resizedImages[0].cols;

  // Allocate memory for the batch blob: batch * channels * height * width
  size_t totalSize =
      batchSize * 3 * resizedImages[0].rows * resizedImages[0].cols;
  blob = new float[totalSize];

  // Convert each image to float and normalize, then copy to blob in CHW format
  for (size_t b = 0; b < batchSize; ++b) {
    cv::Mat floatImage;
    resizedImages[b].convertTo(floatImage, CV_32FC3, 1 / 255.0f);

    std::vector<cv::Mat> chw(3);
    for (int c = 0; c < 3; ++c) {
      chw[c] =
          cv::Mat(resizedImages[b].rows, resizedImages[b].cols, CV_32FC1,
                  blob + b * 3 * resizedImages[b].rows * resizedImages[b].cols +
                      c * resizedImages[b].rows * resizedImages[b].cols);
    }
    cv::split(floatImage, chw);
  }
#ifdef DEBUG_MODE
  DEBUG_PRINT("Batch preprocessing completed");
#endif
  return resizedShapes;
}
// Postprocess function to convert raw model output into detections
std::vector<Detection> YOLODetector::postprocess(
    const cv::Size &originalImageSize, const cv::Size &resizedImageShape,
    const std::vector<Ort::Value> &outputTensors, int img_idx,
    float confThreshold, float iouThreshold) {
#ifdef TIMING_MODE
  ScopedTimer timer("postprocessing"); // Measure postprocessing time
#endif
  std::vector<Detection> detections;
  const float *rawOutput =
      outputTensors[0].GetTensorData<float>(); // Extract raw output data from
                                               // the first output tensor
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

  int det_offset = img_idx * outputShape[1] * outputShape[2];
  // Determine the number of features and detections
  const size_t num_features = outputShape[1];
  const size_t num_detections = outputShape[2];
  // Early exit if no detections
  if (num_detections == 0) {
    return detections;
  }
  // Calculate number of classes based on output shape
  const int numClasses = static_cast<int>(num_features) - 4;
  if (numClasses <= 0) {
    // Invalid number of classes
    return detections;
  }
  // Reserve memory for efficient appending
  std::vector<BoundingBox> boxes;
  boxes.reserve(num_detections);
  std::vector<float> confs;
  confs.reserve(num_detections);
  std::vector<int> classIds;
  classIds.reserve(num_detections);
  std::vector<BoundingBox> nms_boxes;
  nms_boxes.reserve(num_detections);
  // Constants for indexing
  const float *ptr = rawOutput;

  for (size_t d = 0; d < num_detections; ++d) {
    // Extract bounding box coordinates (center x, center y, width, height)
    float centerX = ptr[det_offset + 0 * num_detections + d];
    float centerY = ptr[det_offset + 1 * num_detections + d];
    float width = ptr[det_offset + 2 * num_detections + d];
    float height = ptr[det_offset + 3 * num_detections + d];
    // Find class with the highest confidence score
    int classId = -1;
    float maxScore = -FLT_MAX;
    for (int c = 0; c < numClasses; ++c) {
      const float score = ptr[det_offset + d + (4 + c) * num_detections];
      if (score > maxScore) {
        maxScore = score;
        classId = c;
      }
    }
    // Proceed only if confidence exceeds threshold
    if (maxScore > confThreshold) {
      // Convert center coordinates to top-left (x1, y1)
      float left = centerX - width / 2.0f;
      float top = centerY - height / 2.0f;

      // Scale to original image size
      BoundingBox scaledBox = utils::ImagePreprocessingUtils::scaleCoords(
          resizedImageShape, BoundingBox(left, top, width, height),
          originalImageSize, true);

      // Round coordinates for integer pixel positions
      BoundingBox roundedBox;
      roundedBox.x = std::round(scaledBox.x);
      roundedBox.y = std::round(scaledBox.y);
      roundedBox.width = std::round(scaledBox.width);
      roundedBox.height = std::round(scaledBox.height);

      // Adjust NMS box coordinates to prevent overlap between classes
      BoundingBox nmsBox = roundedBox;
      nmsBox.x += classId * 7680; // Arbitrary offset to differentiate classes
      nmsBox.y += classId * 7680;

      // Add to respective containers
      nms_boxes.emplace_back(nmsBox);
      boxes.emplace_back(roundedBox);
      confs.emplace_back(maxScore);
      classIds.emplace_back(classId);
    }
  }
  // Apply Non-Maximum Suppression (NMS) to eliminate redundant detections
  std::vector<int> indices;
  utils::NMSBoxes(nms_boxes, confs, confThreshold, iouThreshold, indices);

  // Collect filtered detections into the result vector
  detections.reserve(indices.size());
  for (const int idx : indices) {
    detections.emplace_back(Detection{
        boxes[idx],   // Bounding box
        confs[idx],   // Confidence score
        classIds[idx] // Class ID
    });
  }
#ifdef DEBUG_MODE
  DEBUG_PRINT("Postprocessing completed") // Debug log for completion
#endif
  return detections;
}
// Postprocess function implementation
std::vector<Detection> YOLODetector::postprocess_yolo10(
    const cv::Size &originalImageSize, const cv::Size &resizedImageShape,
    const std::vector<Ort::Value> &outputTensors, int img_idx,
    float confThreshold, float iouThreshold) {
  // Start timing the postprocessing step
#ifdef TIMING_MODE
  ScopedTimer timer("Postprocessing");
#endif
  std::vector<Detection> detections;
  // Retrieve raw output data from the first output tensor
  auto *rawOutput = outputTensors[0].GetTensorData<float>();
  std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

  std::vector<Detection> detectionVector;
  int det_offset = img_idx * outputShape[1] * outputShape[2];
  // Assume the second dimension represents the number of detections
  int num_detections = outputShape[1];
  if (num_detections == 0)
    return detections;
#ifdef DEBUG_MODE
  DEBUG_PRINT("Number of detections before filtering: " << num_detections);
#endif
  // Reserve memory for efficient appending
  std::vector<BoundingBox> boxes;
  boxes.reserve(num_detections);
  std::vector<float> confs;
  confs.reserve(num_detections);
  std::vector<int> classIds;
  classIds.reserve(num_detections);
  std::vector<BoundingBox> nms_boxes;
  nms_boxes.reserve(num_detections);
  // Iterate through each detection and filter based on confidence threshold
  for (int i = 0; i < num_detections; i++) {
    float x1 = rawOutput[det_offset + i * 6 + 0];
    float y1 = rawOutput[det_offset + i * 6 + 1];
    float x2 = rawOutput[det_offset + i * 6 + 2];
    float y2 = rawOutput[det_offset + i * 6 + 3];
    float confidence = rawOutput[det_offset + i * 6 + 4];
    int classId = static_cast<int>(rawOutput[det_offset + i * 6 + 5]);

    // Proceed only if confidence exceeds threshold
    if (confidence > confThreshold) {
      // Scale to original image size
      BoundingBox scaledBox = utils::ImagePreprocessingUtils::scaleCoords(
          resizedImageShape, BoundingBox(x1, y1, x2 - x1, y2 - y1),
          originalImageSize, true);

      // Round coordinates for integer pixel positions
      BoundingBox roundedBox;
      roundedBox.x = std::round(scaledBox.x);
      roundedBox.y = std::round(scaledBox.y);
      roundedBox.width = std::round(scaledBox.width);
      roundedBox.height = std::round(scaledBox.height);

      // Adjust NMS box coordinates to prevent overlap between classes
      BoundingBox nmsBox = roundedBox;
      nmsBox.x += classId * 7680; // Arbitrary offset to differentiate classes
      nmsBox.y += classId * 7680;

      // Add to respective containers
      nms_boxes.emplace_back(nmsBox);
      boxes.emplace_back(roundedBox);
      confs.emplace_back(confidence);
      classIds.emplace_back(classId);
    }
  }

  // Apply Non-Maximum Suppression (NMS) to eliminate redundant detections
  std::vector<int> indices;
  utils::NMSBoxes(nms_boxes, confs, confThreshold, iouThreshold, indices);

  // Collect filtered detections into the result vector
  detections.reserve(indices.size());
  for (const int idx : indices) {
    detections.emplace_back(Detection{
        boxes[idx],   // Bounding box
        confs[idx],   // Confidence score
        classIds[idx] // Class ID
    });
  }
#ifdef DEBUG_MODE
  DEBUG_PRINT("Postprocessing completed") // Debug log for completion
#endif
  return detections;
}

// Postprocess function implementation
std::vector<Detection> YOLODetector::postprocess_yolonas(
    const cv::Size &originalImageSize, const cv::Size &resizedImageShape,
    const std::vector<Ort::Value> &outputTensors, int img_idx,
    float confThreshold, float iouThreshold) {
  // Start timing the postprocessing step
#ifdef TIMING_MODE
  ScopedTimer timer("Postprocessing");
#endif
  std::vector<Detection> detections;
  // Retrieve raw output data from the first output tensor
  auto *rawOutput = outputTensors[0].GetTensorData<float>();
  auto *rawOutput1 = outputTensors[1].GetTensorData<float>();
  std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> outputShape1 =
      outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();

  std::vector<Detection> detectionVector;
  int det_offset = img_idx * outputShape[1] * outputShape[2];
  int det_offset1 = img_idx * outputShape1[1] * outputShape1[2];
  // Assume the second dimension represents the number of detections
  int num_detections = outputShape[1];
#ifdef DEBUG_MODE
  DEBUG_PRINT("Number of detections before filtering: " << num_detections);
#endif
  std::cout << "Number of detections before filtering: " << num_detections
            << std::endl;
  if (num_detections == 0)
    return detections;
#ifdef DEBUG_MODE
  DEBUG_PRINT("Number of detections before filtering: " << num_detections);
#endif
  // Reserve memory for efficient appending
  std::vector<BoundingBox> boxes;
  boxes.reserve(num_detections);
  std::vector<float> confs;
  confs.reserve(num_detections);
  std::vector<int> classIds;
  classIds.reserve(num_detections);
  std::vector<BoundingBox> nms_boxes;
  nms_boxes.reserve(num_detections);

  // Iterate through each detection and filter based on confidence threshold
  for (int i = 0; i < num_detections; i++) {
    float x1 = rawOutput[det_offset + i * 4 + 0];
    float y1 = rawOutput[det_offset + i * 4 + 1];
    float x2 = rawOutput[det_offset + i * 4 + 2];
    float y2 = rawOutput[det_offset + i * 4 + 3];
    int classId = -1;
    float confidence = -FLT_MAX;
    for (int c = 0; c < outputShape1[2]; ++c) {
      const float score = rawOutput1[det_offset1 + i * outputShape1[2] + c];
      if (score > confidence) {
        confidence = score;
        classId = c;
      }
    }

    // Proceed only if confidence exceeds threshold
    if (confidence > confThreshold) {
      // Scale to original image size
      BoundingBox scaledBox = utils::ImagePreprocessingUtils::scaleCoords(
          resizedImageShape, BoundingBox(x1, y1, x2 - x1, y2 - y1),
          originalImageSize, true);

      // Round coordinates for integer pixel positions
      BoundingBox roundedBox;
      roundedBox.x = std::round(scaledBox.x);
      roundedBox.y = std::round(scaledBox.y);
      roundedBox.width = std::round(scaledBox.width);
      roundedBox.height = std::round(scaledBox.height);

      // Adjust NMS box coordinates to prevent overlap between classes
      BoundingBox nmsBox = roundedBox;
      nmsBox.x += classId * 7680; // Arbitrary offset to differentiate classes
      nmsBox.y += classId * 7680;

      // Add to respective containers
      nms_boxes.emplace_back(nmsBox);
      boxes.emplace_back(roundedBox);
      confs.emplace_back(confidence);
      classIds.emplace_back(classId);
    }
  }

  // Apply Non-Maximum Suppression (NMS) to eliminate redundant detections
  std::vector<int> indices;
  utils::NMSBoxes(nms_boxes, confs, confThreshold, iouThreshold, indices);

  // Collect filtered detections into the result vector
  detections.reserve(indices.size());
  for (const int idx : indices) {
    detections.emplace_back(Detection{
        boxes[idx],   // Bounding box
        confs[idx],   // Confidence score
        classIds[idx] // Class ID
    });
  }
#ifdef DEBUG_MODE
  DEBUG_PRINT("Postprocessing completed") // Debug log for completion
#endif
  return detections;
}

// Postprocess function implementation
std::vector<Detection> YOLODetector::postprocess_yolo7(
    const cv::Size &originalImageSize, const cv::Size &resizedImageShape,
    const std::vector<Ort::Value> &outputTensors, int img_idx,
    float confThreshold, float iouThreshold) {
#ifdef TIMING_MODE
  ScopedTimer timer("postprocessing"); // Measure postprocessing time
#endif
  std::vector<Detection> detections;
  const float *rawOutput =
      outputTensors[0].GetTensorData<float>(); // Extract raw output data from
                                               // the first output tensor
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

  int det_offset = img_idx * outputShape[1] * outputShape[2];
  // Determine the number of features and detections
  const size_t num_features = outputShape[2];
  const size_t num_detections = outputShape[1];
  // Early exit if no detections
  if (num_detections == 0) {
    return detections;
  }
  // Calculate number of classes based on output shape
  const int numClasses = static_cast<int>(num_features) - 5;
  if (numClasses <= 0) {
    // Invalid number of classes
    return detections;
  }
  // Reserve memory for efficient appending
  std::vector<BoundingBox> boxes;
  boxes.reserve(num_detections);
  std::vector<float> confs;
  confs.reserve(num_detections);
  std::vector<int> classIds;
  classIds.reserve(num_detections);
  std::vector<BoundingBox> nms_boxes;
  nms_boxes.reserve(num_detections);
  // Constants for indexing
  const float *ptr = rawOutput;

  for (size_t d = 0; d < num_detections; ++d) {
    // Extract bounding box coordinates (center x, center y, width, height)
    float centerX = ptr[det_offset + d * num_features + 0];
    float centerY = ptr[det_offset + d * num_features + 1];
    float width = ptr[det_offset + d * num_features + 2];
    float height = ptr[det_offset + d * num_features + 3];
    // Find class with the highest confidence score
    float obj = ptr[det_offset + d * num_features + 4];
    int classId = -1;
    float maxScore = -FLT_MAX;
    for (int c = 0; c < numClasses; ++c) {
      const float score = ptr[det_offset + (5 + c) + d * num_features];
      if (score > maxScore) {
        maxScore = score;
        classId = c;
      }
    }

    // Proceed only if confidence exceeds threshold
    if (obj > confThreshold) {
      // Convert center coordinates to top-left (x1, y1)
      float left = centerX - width / 2.0f;
      float top = centerY - height / 2.0f;

      // Scale to original image size
      BoundingBox scaledBox = utils::ImagePreprocessingUtils::scaleCoords(
          resizedImageShape, BoundingBox(left, top, width, height),
          originalImageSize, true);

      // Round coordinates for integer pixel positions
      BoundingBox roundedBox;
      roundedBox.x = std::round(scaledBox.x);
      roundedBox.y = std::round(scaledBox.y);
      roundedBox.width = std::round(scaledBox.width);
      roundedBox.height = std::round(scaledBox.height);

      // Adjust NMS box coordinates to prevent overlap between classes
      BoundingBox nmsBox = roundedBox;
      nmsBox.x += classId * 7680; // Arbitrary offset to differentiate classes
      nmsBox.y += classId * 7680;

      // Add to respective containers
      nms_boxes.emplace_back(nmsBox);
      boxes.emplace_back(roundedBox);
      confs.emplace_back(obj);
      classIds.emplace_back(classId);
    }
  }
  // Apply Non-Maximum Suppression (NMS) to eliminate redundant detections
  std::vector<int> indices;
  utils::NMSBoxes(nms_boxes, confs, confThreshold, iouThreshold, indices);

  // Collect filtered detections into the result vector
  detections.reserve(indices.size());
  for (const int idx : indices) {
    detections.emplace_back(Detection{
        boxes[idx],   // Bounding box
        confs[idx],   // Confidence score
        classIds[idx] // Class ID
    });
  }
#ifdef DEBUG_MODE
  DEBUG_PRINT("Postprocessing completed") // Debug log for completion
#endif
  return detections;
}

// Detect function implementation
std::vector<Detection> YOLODetector::detect(const cv::Mat &image,
                                            float confThreshold,
                                            float iouThreshold) {
#ifdef TIMING_MODE
  ScopedTimer timer("Overall detection");
#endif
  float *blobPtr = nullptr; // Pointer to hold preprocessed image data
  // Define the shape of the input tensor (batch size, channels, height, width)
  std::vector<int64_t> inputTensorShape = {1, 3, inputImageShape.height,
                                           inputImageShape.width};
  auto start = std::chrono::high_resolution_clock::now();
  // Preprocess the image and obtain a pointer to the blob
  cv::Mat preprocessedImage = preprocess(image, blobPtr, inputTensorShape);
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start);

  // Compute the total number of elements in the input tensor
  size_t inputTensorSize = utils::MathUtils::vectorProduct(inputTensorShape);

  // Create a vector from the blob data for ONNX Runtime input
  std::vector<float> inputTensorValues(blobPtr, blobPtr + inputTensorSize);

  delete[] blobPtr; // Free the allocated memory for the blob

  // Create an Ort memory info object (can be cached if used repeatedly)
  static Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Create input tensor object using the preprocessed data
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensorValues.data(), inputTensorSize,
      inputTensorShape.data(), inputTensorShape.size());
  // std::cout << "data " << outputNames.data()[0] << std::endl;
  // std::cout << "preprocessing completed in: " << duration.count() << " ms" <<
  // std::endl;

  start = std::chrono::high_resolution_clock::now();
  // Run the inference session with the input tensor and retrieve output tensors
  std::vector<Ort::Value> outputTensors =
      session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                  numInputNodes, outputNames.data(), numOutputNodes);
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start);
  // std::cout << "inference completed in: " << duration.count() << " ms" <<
  // std::endl; Determine the resized image shape based on input tensor shape
  start = std::chrono::high_resolution_clock::now();
  cv::Size resizedImageShape(static_cast<int>(inputTensorShape[3]),
                             static_cast<int>(inputTensorShape[2]));
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start);
  // std::cout << "resizing completed in: " << duration.count() << " ms" <<
  // std::endl; Postprocess the output tensors to obtain detections
  start = std::chrono::high_resolution_clock::now();
  std::vector<Detection> detections;
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  // std::cout << "outputShape " << outputShape[0] << " " << outputShape[1] << "
  // " << outputShape[2] << " " << outputShape[3] << std::endl;
  if (outputShape[2] == 6) {
    // std::cout << "yolo 10 detected" << std::endl;
    detections =
        postprocess_yolo10(image.size(), resizedImageShape, outputTensors, 0,
                           confThreshold, iouThreshold);
  } else if (outputShape[2] == 4) {
    // std::cout << "yolo nas detected" << std::endl;
    detections =
        postprocess_yolonas(image.size(), resizedImageShape, outputTensors, 0,
                            confThreshold, iouThreshold);
  } else if (outputShape[1] > outputShape[2]) {
    detections =
        postprocess_yolo7(image.size(), resizedImageShape, outputTensors, 0,
                          confThreshold, iouThreshold);
  } else {
    // std::cout << "yolo not 10 detected" << std::endl;
    detections = postprocess(image.size(), resizedImageShape, outputTensors, 0,
                             confThreshold, iouThreshold);
  }
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start);
  // std::cout << "postporcessing completed in: " << duration.count() << " ms"
  // << std::endl;
  return detections; // Return the vector of detections
}

// Batch detect function implementation
std::vector<std::vector<Detection>>
YOLODetector::detect(const std::vector<cv::Mat> &images, float confThreshold,
                     float iouThreshold) {
#ifdef TIMING_MODE
  ScopedTimer timer("Overall batch detection");
#endif
  if (images.empty()) {
    return {};
  }

  float *blobPtr = nullptr; // Pointer to hold preprocessed batch data
  // Define the shape of the input tensor (batch size, channels, height, width)
  std::vector<int64_t> inputTensorShape = {static_cast<int64_t>(images.size()),
                                           3, inputImageShape.height,
                                           inputImageShape.width};
  auto start = std::chrono::high_resolution_clock::now();
  // Batch preprocess the images and obtain a pointer to the blob
  std::vector<cv::Size> resizedShapes =
      batch_preprocess(images, blobPtr, inputTensorShape);
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start);

  // Compute the total number of elements in the input tensor
  size_t inputTensorSize = utils::MathUtils::vectorProduct(inputTensorShape);

  // Create a vector from the blob data for ONNX Runtime input
  std::vector<float> inputTensorValues(blobPtr, blobPtr + inputTensorSize);

  delete[] blobPtr; // Free the allocated memory for the blob

  // Create an Ort memory info object (can be cached if used repeatedly)
  static Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // Create input tensor object using the preprocessed data
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensorValues.data(), inputTensorSize,
      inputTensorShape.data(), inputTensorShape.size());
  std::cout << "batch preprocessing completed in: " << duration.count() << " ms"
            << std::endl;
  start = std::chrono::high_resolution_clock::now();
  // Run the inference session with the input tensor and retrieve output tensors
  std::vector<Ort::Value> outputTensors =
      session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                  numInputNodes, outputNames.data(), numOutputNodes);
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start);
  std::cout << "batch inference completed in: " << duration.count() << " ms"
            << std::endl;

  // Determine the resized image shape based on input tensor shape
  start = std::chrono::high_resolution_clock::now();
  cv::Size resizedImageShape(static_cast<int>(inputTensorShape[3]),
                             static_cast<int>(inputTensorShape[2]));
  std::vector<std::vector<Detection>> alldetections;
  std::vector<Detection> detections;

  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  std::cout << "outputShape " << outputShape[0] << " " << outputShape[1] << " "
            << outputShape[2] << " " << outputShape[3] << std::endl;
  int batch_size = outputShape[0];
  cv::Mat image;
  for (size_t i = 0; i < batch_size; i++) {
    image = images[i];
    if (outputShape[2] == 6) {
      // std::cout << "yolo 10 detected" << std::endl;
      detections =
          postprocess_yolo10(image.size(), resizedImageShape, outputTensors, i,
                             confThreshold, iouThreshold);
    } else if (outputShape[2] == 4) {
      // std::cout << "yolo nas detected" << std::endl;
      detections =
          postprocess_yolonas(image.size(), resizedImageShape, outputTensors, i,
                              confThreshold, iouThreshold);
    } else if (outputShape[1] > outputShape[2]) {
      detections =
          postprocess_yolo7(image.size(), resizedImageShape, outputTensors, i,
                            confThreshold, iouThreshold);
    } else {
      // std::cout << "yolo not 10 detected" << std::endl;
      detections = postprocess(image.size(), resizedImageShape, outputTensors,
                               i, confThreshold, iouThreshold);
    }
    alldetections.push_back(detections);
  }

  duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start);
  std::cout << "postporcessing completed in: " << duration.count() << " ms"
            << std::endl;
  return alldetections; // Return the vector of detections
}
