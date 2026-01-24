#include "yolos/core/utils.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

namespace yolos {
namespace utils {

size_t vectorProduct(const std::vector<int64_t> &shape) {
  if (shape.empty())
    return 0;
  return std::accumulate(shape.begin(), shape.end(), 1ULL,
                         std::multiplies<size_t>());
}

std::vector<std::string> getClassNames(const std::string &path) {
  std::vector<std::string> classNames;
  std::ifstream infile(path);

  if (!infile) {
    std::cerr << "[ERROR] Failed to open class names file: " << path
              << std::endl;
    return classNames;
  }

  std::string line;
  while (std::getline(infile, line)) {
    // Remove carriage return if present (Windows compatibility)
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      classNames.emplace_back(line);
    }
  }

  return classNames;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void sigmoidInplace(std::vector<float> &values) {
  for (auto &v : values) {
    v = sigmoid(v);
  }
}

} // namespace utils
} // namespace yolos
