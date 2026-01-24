#pragma once

// ============================================================================
// YOLO Core Utilities
// ============================================================================
// Common utility functions used across all YOLO tasks.
// All functions are marked inline to prevent ODR violations.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace yolos {
namespace utils {

// ============================================================================
// Math Utilities
// ============================================================================

/// @brief Clamp a value to a specified range [low, high]
/// @tparam T Arithmetic type (int, float, etc.)
/// @param value The value to clamp
/// @param low Lower bound
/// @param high Upper bound
/// @return Clamped value
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
clamp(const T &value, const T &low, const T &high) {
  // Ensure range is valid; swap if necessary
  T validLow = low < high ? low : high;
  T validHigh = low < high ? high : low;

  if (value < validLow)
    return validLow;
  if (value > validHigh)
    return validHigh;
  return value;
}

/// @brief Compute the product of elements in a vector
/// @param shape Vector of dimensions
/// @return Product of all elements
size_t vectorProduct(const std::vector<int64_t> &shape);

// ============================================================================
// File I/O Utilities
// ============================================================================

/// @brief Load class names from a file (one class name per line)
/// @param path Path to the class names file
/// @return Vector of class names
std::vector<std::string> getClassNames(const std::string &path);

// ============================================================================
// Sigmoid Activation
// ============================================================================

/// @brief Apply sigmoid activation: 1 / (1 + exp(-x))
/// @param x Input value
/// @return Sigmoid of x
float sigmoid(float x);

/// @brief Apply sigmoid activation to a vector in-place
/// @param values Vector of values to transform
void sigmoidInplace(std::vector<float> &values);

} // namespace utils
} // namespace yolos
