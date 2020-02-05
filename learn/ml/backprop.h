#pragma once

#include <array>
#include <tuple>

#include "learn/ml/elementwise_subtraction.h"

namespace ml {

template <typename T, size_t X, size_t H>
std::tuple<
    std::array<T, X * H>, std::array<T, X * H>, std::array<T, X>, std::array<T, X>, >
BackPropagate(const std::array<T, X * H>& W1, const std::array<T, X>& b1,
              const std::array<T, X * H>& W2, const std::array<T, X>& b2,
              const std::array<T, X>& Z1, const std::array<T, X>& A1,
              const std::array<T, X>& A2, const std::array<T, X>& input,
              const std::array<T, X>& output, size_t batchSize, ) {
  // TODO:

  // Error at last error.
  std::array<T, X> dZ2 = ElementwiseSubtraction(A2, Y);

  // Gradients at last layer.
  std::array<T, X> dW2 = ScalarMatrixMultiplication(
      1 / batchSize, MatrixVectorMultiplication(dZ2, Transpose(A1)));
  std::array<T, X> db2 =
      ScalarVectorMultiplication(1 / batchSize, MatrixRowSum(dZ2));

  // Back propagate through the first layer.
  std::array<T, X> dA1 = MatrixVectorMultiplication(Transpose(W2), dZ2);
  std::array<T, X> dZ1 = dA1 * Sigmoide(Z1) * (1 - Sigmoid(Z1));

  // Gradients at first layer.
  std::array<T, X> dW1 = ScalarMatrixMultiplication(
      1 / batchSize, MatrixVectorMultiplication(dZ1, Transpose(input)));
  std::array<T, X> db1 =
      ScalarVectorMultiplication(1 / batchSize, MatrixRowSum(dZ1));

  return { dW1, db1, dW2, db2 }
}

}  // namespace ml
