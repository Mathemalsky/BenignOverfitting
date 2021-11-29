#include "mnist.hpp"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

// headers from Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "constants.hpp"
#include "io.hpp"
#include "types.hpp"

using Matrix_X = Eigen::Matrix<float, MNIST::PIXELS_PER_IMAGE, Eigen::Dynamic>;
using Vector_Y = Eigen::Matrix<float, Eigen::Dynamic, MNIST::POSSIBLE_LABELS>;

namespace MNIST {
Matrix_X buildMatrix_X(const size_t n) {
  Data images = readTrainImages(n);
  Matrix_X x(PIXELS_PER_IMAGE, n);
  assert((images.size = PIXELS_PER_IMAGE * n) && "Number of bytes read from file does not fit.");

  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < PIXELS_PER_IMAGE; ++i) {
      x(i, j) = (float) images.data[PIXELS_PER_IMAGE * j + i] / MAXBYTE;
    }
  }
  free(images.data);
  return x;
}

Vector_Y buildVector_Y(const size_t n) {
  Data labels = readTrainLables(n);
  assert(labels.size == n && "Number of labels read and n are missmatching!");

  Vector_Y y(n, POSSIBLE_LABELS);

  for (size_t i = 0; i < n; ++i) {
    for (unsigned char j = 0; j < POSSIBLE_LABELS; ++j) {
      y(i, j) = (j == labels.data[i]) ? 1.0f : 0.f;
    }
  }
  free(labels.data);
  return y;
}

void mnist(int argc, char* argv[]) {
  if (argc < 3) {
    throw std::runtime_error("Too less arguments!");
  }
  const size_t n = atoi(argv[2]);
  std::cerr << buildMatrix_X(n).transpose() << "\n";
  std::cerr << buildVector_Y(n) << "\n";
}
}  // namespace MNIST
