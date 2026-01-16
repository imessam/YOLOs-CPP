#include "utils/YOLO-Utils.hpp"
#include <cmath>
#include <fstream>
#include <random>

namespace utils {

namespace MathUtils {
size_t vectorProduct(const std::vector<int64_t> &vector) {
  if (vector.empty())
    return 0;
  size_t product = 1;
  for (const auto &element : vector)
    product *= element;
  return product;
}
} // namespace MathUtils

namespace ImagePreprocessingUtils {
void letterBox(const cv::Mat &image, cv::Mat &outImage,
               const cv::Size &newShape, const cv::Scalar &color, bool auto_,
               bool scaleFill, bool scaleUp, int stride) {
  cv::Size shape = image.size();
  float r = std::min((float)newShape.height / (float)shape.height,
                     (float)newShape.width / (float)shape.width);
  if (!scaleUp)
    r = std::min(r, 1.0f);

  float ratio[2]{r, r};
  int newUnpad[2]{(int)std::round((float)shape.width * r),
                  (int)std::round((float)shape.height * r)};

  auto dw = (float)(newShape.width - newUnpad[0]);
  auto dh = (float)(newShape.height - newUnpad[1]);

  if (auto_) {
    dw = (float)((int)dw % stride);
    dh = (float)((int)dh % stride);
  } else if (scaleFill) {
    dw = 0.0f;
    dh = 0.0f;
    newUnpad[0] = newShape.width;
    newUnpad[1] = newShape.height;
    ratio[0] = (float)newShape.width / (float)shape.width;
    ratio[1] = (float)newShape.height / (float)shape.height;
  }

  dw /= 2.0f;
  dh /= 2.0f;

  if (shape.width != newUnpad[0] || shape.height != newUnpad[1]) {
    cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]), 0, 0,
               cv::INTER_LINEAR);
  } else {
    outImage = image.clone();
  }

  int top = (int)std::round(dh - 0.1f);
  int bottom = (int)std::round(dh + 0.1f);
  int left = (int)std::round(dw - 0.1f);
  int right = (int)std::round(dw + 0.1f);

  cv::copyMakeBorder(outImage, outImage, top, bottom, left, right,
                     cv::BORDER_CONSTANT, color);
}

BoundingBox scaleCoords(const cv::Size &imageShape, BoundingBox coords,
                        const cv::Size &imageOriginalShape, bool p_Clip) {
  float gain =
      std::min((float)imageShape.height / (float)imageOriginalShape.height,
               (float)imageShape.width / (float)imageOriginalShape.width);

  int pad[2];
  pad[0] =
      (int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) /
            2.0f);
  pad[1] = (int)(((float)imageShape.height -
                  (float)imageOriginalShape.height * gain) /
                 2.0f);

  coords.x = (int)std::round(((float)coords.x - (float)pad[0]) / gain);
  coords.y = (int)std::round(((float)coords.y - (float)pad[1]) / gain);
  coords.width = (int)std::round((float)coords.width / gain);
  coords.height = (int)std::round((float)coords.height / gain);

  if (p_Clip) {
    coords.x = MathUtils::clamp(coords.x, 0, imageOriginalShape.width);
    coords.y = MathUtils::clamp(coords.y, 0, imageOriginalShape.height);
    coords.width =
        MathUtils::clamp(coords.width, 0, imageOriginalShape.width - coords.x);
    coords.height = MathUtils::clamp(coords.height, 0,
                                     imageOriginalShape.height - coords.y);
  }

  return coords;
}
} // namespace ImagePreprocessingUtils

namespace DrawingUtils {
std::vector<cv::Scalar>
generateColors(const std::vector<std::string> &classNames, int seed) {
  std::vector<cv::Scalar> colors;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dis(0, 255);
  for (size_t i = 0; i < classNames.size(); ++i) {
    colors.emplace_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));
  }
  return colors;
}

void drawBoundingBox(cv::Mat &image, const std::vector<Detection> &results,
                     const std::vector<std::string> &classNames,
                     const std::vector<cv::Scalar> &classColors,
                     float maskAlpha) {
  for (const auto &res : results) {
    cv::Scalar color = classColors[res.classId % classColors.size()];
    cv::rectangle(image,
                  cv::Rect(res.box.x, res.box.y, res.box.width, res.box.height),
                  color, 2);

    // Draw label
    std::string label =
        classNames[res.classId] + " " + std::to_string(res.conf).substr(0, 4);
    int baseLine;
    cv::Size labelSize =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    cv::rectangle(image,
                  cv::Rect(res.box.x, res.box.y - labelSize.height,
                           labelSize.width, labelSize.height + baseLine),
                  color, cv::FILLED);
    cv::putText(image, label, cv::Point(res.box.x, res.box.y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    // Draw mask if present
    if (!res.mask.empty()) {
      cv::Mat coloredMask;
      cv::Mat maskBGR;
      cv::cvtColor(res.mask, maskBGR, cv::COLOR_GRAY2BGR);

      // Create a color image from the mask
      cv::Mat coloredPart(res.mask.size(), CV_8UC3, color);
      cv::bitwise_and(coloredPart, maskBGR, coloredMask);

      // Apply mask to region of interest
      cv::Rect roiRect(res.box.x, res.box.y, res.box.width, res.box.height);
      // Ensure ROI is within image boundaries
      roiRect &= cv::Rect(0, 0, image.cols, image.rows);

      if (roiRect.width > 0 && roiRect.height > 0) {
        cv::Mat roi = image(roiRect);
        cv::addWeighted(
            roi, 1.0 - maskAlpha,
            coloredMask(cv::Rect(0, 0, roiRect.width, roiRect.height)),
            maskAlpha, 0, roi);
      }
    }

    // Draw keypoints if present
    for (const auto &kp : res.keypoints) {
      if (kp.conf > 0.5f) {
        cv::circle(image, cv::Point((int)kp.x, (int)kp.y), 3,
                   cv::Scalar(0, 255, 0), cv::FILLED);
      }
    }
  }
}
} // namespace DrawingUtils

std::vector<std::string> getClassNames(const std::string &path) {
  std::vector<std::string> classNames;
  std::ifstream infile(path);
  if (infile) {
    std::string line;
    while (getline(infile, line)) {
      if (!line.empty() && line.back() == '\r')
        line.pop_back();
      classNames.emplace_back(line);
    }
  }
  return classNames;
}

void NMSBoxes(const std::vector<BoundingBox> &boundingBoxes,
              const std::vector<float> &scores, float scoreThreshold,
              float nmsThreshold, std::vector<int> &indices) {
  indices.clear();
  std::vector<int> sorted_indices;
  for (int i = 0; i < (int)boundingBoxes.size(); ++i) {
    if (scores[i] >= scoreThreshold)
      sorted_indices.push_back(i);
  }

  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&](int i, int j) { return scores[i] > scores[j]; });

  std::vector<bool> keep((int)boundingBoxes.size(), true);
  for (size_t i = 0; i < sorted_indices.size(); ++i) {
    int idx1 = sorted_indices[i];
    if (!keep[idx1])
      continue;
    indices.push_back(idx1);

    for (size_t j = i + 1; j < sorted_indices.size(); ++j) {
      int idx2 = sorted_indices[j];
      if (!keep[idx2])
        continue;

      BoundingBox intersection =
          boundingBoxes[idx1].intersect(boundingBoxes[idx2]);
      float inter_area = intersection.area();
      float union_area =
          boundingBoxes[idx1].area() + boundingBoxes[idx2].area() - inter_area;
      if (inter_area / union_area > nmsThreshold)
        keep[idx2] = false;
    }
  }
}

cv::Mat sigmoid(const cv::Mat &src) {
  cv::Mat dst;
  cv::exp(-src, dst);
  return 1.0 / (1.0 + dst);
}

} // namespace utils
