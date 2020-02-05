#pragma once

namespace ml {

// Compute elementwise A + B and return a new array.
template <typename T, size_t N>
std::array<T, N> ElementwiseAddition(const std::array<T, N>& A,
                                     const std::array<T, N>& B) {
  std::array<T, N> out;
  for (size_t i = 0; i < N; ++i) {
    out[i] = A[i] + B[i];
  }

  return out;
}

// Compute elementwise A + B and store the result in B.
template <typename T, size_t N>
void ElementwiseAddition(const std::array<T, N>& A, std::array<T, N>* B) {
  for (size_t i = 0; i < N; ++i) {
    (*B)[i] = A[i] + (*B)[i];
  }
}

}  // namespace ml
