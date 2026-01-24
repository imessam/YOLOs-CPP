#include "yolos/core/types.hpp"
#include <algorithm>

namespace yolos {

float BoundingBox::area() const noexcept {
  return static_cast<float>(width * height);
}

BoundingBox BoundingBox::intersect(const BoundingBox &other) const noexcept {
  int xStart = std::max(x, other.x);
  int yStart = std::max(y, other.y);
  int xEnd = std::min(x + width, other.x + other.width);
  int yEnd = std::min(y + height, other.y + other.height);
  int iw = std::max(0, xEnd - xStart);
  int ih = std::max(0, yEnd - yStart);
  return BoundingBox(xStart, yStart, iw, ih);
}

float BoundingBox::iou(const BoundingBox &other) const noexcept {
  BoundingBox inter = intersect(other);
  float interArea = inter.area();
  float unionArea = area() + other.area() - interArea;
  return (unionArea > 0.0f) ? (interArea / unionArea) : 0.0f;
}

float OrientedBoundingBox::area() const noexcept { return width * height; }

const std::vector<std::pair<int, int>> &getPoseSkeleton() {
  static const std::vector<std::pair<int, int>> POSE_SKELETON = {
      // Face connections
      {0, 1},
      {0, 2},
      {1, 3},
      {2, 4},
      // Head-to-shoulder connections
      {3, 5},
      {4, 6},
      // Arms
      {5, 7},
      {7, 9},
      {6, 8},
      {8, 10},
      // Body
      {5, 6},
      {5, 11},
      {6, 12},
      {11, 12},
      // Legs
      {11, 13},
      {13, 15},
      {12, 14},
      {14, 16}};
  return POSE_SKELETON;
}

} // namespace yolos
