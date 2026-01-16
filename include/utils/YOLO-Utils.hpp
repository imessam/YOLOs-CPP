#pragma once

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Struct to represent a bounding box.
 */
struct BoundingBox {
  int x{0};
  int y{0};
  int width{0};
  int height{0};

  BoundingBox() = default;
  BoundingBox(int x_, int y_, int width_, int height_)
      : x(x_), y(y_), width(width_), height(height_) {}

  float area() const { return static_cast<float>(width * height); }

  BoundingBox intersect(const BoundingBox &other) const {
    int xStart = std::max(x, other.x);
    int yStart = std::max(y, other.y);
    int xEnd = std::min(x + width, other.x + other.width);
    int yEnd = std::min(y + height, other.y + other.height);
    int iw = std::max(0, xEnd - xStart);
    int ih = std::max(0, yEnd - yStart);
    return BoundingBox(xStart, yStart, iw, ih);
  }
};

/**
 * @brief Struct representing a detected keypoint in pose estimation.
 */
struct KeyPoint {
  float x{0};
  float y{0};
  float conf{0};
  KeyPoint(float x_ = 0, float y_ = 0, float conf_ = 0)
      : x(x_), y(y_), conf(conf_) {}
};

/**
 * @brief Struct to represent a detection.
 */
struct Detection {
  BoundingBox box;
  float conf{0.0f};
  int classId{-1};
  std::vector<KeyPoint> keypoints; // For pose estimation
  cv::Mat mask;                    // For segmentation

  Detection() = default;
  Detection(BoundingBox box_, float conf_, int classId_)
      : box(box_), conf(conf_), classId(classId_) {}
};

/**
 * @brief Struct to represent an Oriented bounding box (OBB) in xywhr format.
 */
struct OrientedBoundingBox {
  float x;      // x-coordinate of the center
  float y;      // y-coordinate of the center
  float width;  // width of the box
  float height; // height of the box
  float angle;  // rotation angle in radians

  OrientedBoundingBox() : x(0), y(0), width(0), height(0), angle(0) {}
  OrientedBoundingBox(float x_, float y_, float width_, float height_,
                      float angle_)
      : x(x_), y(y_), width(width_), height(height_), angle(angle_) {}
};

/**
 * @brief Struct to represent a detection with an oriented bounding box.
 */
struct OBB_Detection {
  OrientedBoundingBox box;
  float conf{0.0f};
  int classId{-1};

  OBB_Detection() = default;
  OBB_Detection(const OrientedBoundingBox &box_, float conf_, int classId_)
      : box(box_), conf(conf_), classId(classId_) {}
};

/**
 * @brief Struct to represent classification results.
 */
struct ClassificationResult {
  int classId{-1};
  float confidence{0.0f};
  std::string className{};

  ClassificationResult() = default;
  ClassificationResult(int id, float conf, std::string name)
      : classId(id), confidence(conf), className(std::move(name)) {}
};

namespace utils {

/**
 * @namespace MathUtils
 * @brief Namespace containing mathematical utility functions.
 */
namespace MathUtils {
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
clamp(const T &value, const T &low, const T &high) {
  return std::max(low, std::min(value, high));
}

size_t vectorProduct(const std::vector<int64_t> &vector);
} // namespace MathUtils

/**
 * @namespace ImagePreprocessingUtils
 * @brief Namespace containing utility functions for image preprocessing.
 */
namespace ImagePreprocessingUtils {
void letterBox(const cv::Mat &image, cv::Mat &outImage,
               const cv::Size &newShape,
               const cv::Scalar &color = cv::Scalar(114, 114, 114),
               bool auto_ = true, bool scaleFill = false, bool scaleUp = true,
               int stride = 32);

BoundingBox scaleCoords(const cv::Size &imageShape, BoundingBox coords,
                        const cv::Size &imageOriginalShape, bool p_Clip = true);
} // namespace ImagePreprocessingUtils

/**
 * @namespace DrawingUtils
 * @brief Namespace containing utility functions for drawing detections.
 */
namespace DrawingUtils {
std::vector<cv::Scalar>
generateColors(const std::vector<std::string> &classNames, int seed = 42);

void drawBoundingBox(cv::Mat &image, const std::vector<Detection> &results,
                     const std::vector<std::string> &classNames,
                     const std::vector<cv::Scalar> &classColors,
                     float maskAlpha = 0.4f);

inline void
drawClassificationResult(cv::Mat &image, const ClassificationResult &result,
                         const cv::Point &position = cv::Point(10, 10),
                         const cv::Scalar &textColor = cv::Scalar(0, 255, 0),
                         double fontScaleMultiplier = 0.0008,
                         const cv::Scalar &bgColor = cv::Scalar(0, 0, 0)) {
  if (image.empty() || result.classId == -1)
    return;
  std::ostringstream ss;
  ss << result.className << ": " << std::fixed << std::setprecision(2)
     << result.confidence * 100 << "%";
  std::string text = ss.str();
  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = std::min(image.rows, image.cols) * fontScaleMultiplier;
  if (fontScale < 0.4)
    fontScale = 0.4;
  const int thickness = std::max(1, static_cast<int>(fontScale * 1.8));
  int baseline = 0;
  cv::Size textSize =
      cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
  baseline += thickness;
  cv::Point textPosition = position;
  if (textPosition.x < 0)
    textPosition.x = 0;
  if (textPosition.y < textSize.height)
    textPosition.y = textSize.height + 2;
  cv::Point backgroundTopLeft(textPosition.x,
                              textPosition.y - textSize.height - baseline / 3);
  cv::Point backgroundBottomRight(textPosition.x + textSize.width,
                                  textPosition.y + baseline / 2);
  backgroundTopLeft.x =
      MathUtils::clamp(backgroundTopLeft.x, 0, image.cols - 1);
  backgroundTopLeft.y =
      MathUtils::clamp(backgroundTopLeft.y, 0, image.rows - 1);
  backgroundBottomRight.x =
      MathUtils::clamp(backgroundBottomRight.x, 0, image.cols - 1);
  backgroundBottomRight.y =
      MathUtils::clamp(backgroundBottomRight.y, 0, image.rows - 1);
  cv::rectangle(image, backgroundTopLeft, backgroundBottomRight, bgColor,
                cv::FILLED);
  cv::putText(image, text, cv::Point(textPosition.x, textPosition.y), fontFace,
              fontScale, textColor, thickness, cv::LINE_AA);
}
} // namespace DrawingUtils

std::vector<std::string> getClassNames(const std::string &path);

void NMSBoxes(const std::vector<BoundingBox> &boundingBoxes,
              const std::vector<float> &scores, float scoreThreshold,
              float nmsThreshold, std::vector<int> &indices);

cv::Mat sigmoid(const cv::Mat &src);

} // namespace utils
