#pragma once

#include <array>
#include <cmath>

namespace ml {

// Time: O(n)
// Space: O(n)
template <typename T, size_t n>
std::array<T, n> Softmax(const std::array<T, n>& X) {
  std::array<T, n> out;
  T denominator = 0;

  for (size_t i = 0; i < X.size(); ++i) {
    out[i] = exp(X[i]);
    denominator += out[i];
  }

  for (size_t i = 0; i < X.size(); ++i) {
    out[i] /= denominator;
  }

  return out;
}

}  // namespace ml
