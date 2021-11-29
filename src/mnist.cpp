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

using Matrix_X      = Eigen::Matrix<float, MNIST::PIXELS_PER_IMAGE, Eigen::Dynamic>;
using Matrix_Y      = Eigen::Matrix<float, Eigen::Dynamic, MNIST::POSSIBLE_LABELS>;
using Matrix_theta  = Eigen::Matrix<float, MNIST::PIXELS_PER_IMAGE, MNIST::POSSIBLE_LABELS>;
using VectorLabelsf = Eigen::Matrix<float, 1, MNIST::POSSIBLE_LABELS>;

namespace MNIST {
Matrix_X buildMatrix_X(const size_t n, const Data images) {
  assert((images.size == PIXELS_PER_IMAGE * n) && "Number of bytes read from file does not fit.");

  Matrix_X x(PIXELS_PER_IMAGE, n);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < PIXELS_PER_IMAGE; ++i) {
      x(i, j) = (float) images.data[PIXELS_PER_IMAGE * j + i] / MAXBYTE;
    }
  }

  return x;
}

Matrix_Y buildMatrix_Y(const size_t n, const Data labels) {
  assert(labels.size == n && "Number of labels read and n are missmatching!");

  Matrix_Y y(n, POSSIBLE_LABELS);
  for (size_t i = 0; i < n; ++i) {
    for (unsigned char j = 0; j < POSSIBLE_LABELS; ++j) {
      y(i, j) = (j == labels.data[i]) ? 1.0f : 0.f;
    }
  }
  return y;
}

static size_t maxIndex(const VectorLabelsf vec) {
  assert(vec.cols() != 0 && "Vector should be non empty!");
  size_t index = 0;
  float max    = vec(0);
  for (size_t i = 1; i < vec.cols(); ++i) {
    if (vec(i) > max) {
      max   = vec(i);
      index = i;
    }
  }
  return index;
}

float predict(const size_t t, const Matrix_theta theta) {
  Data images = readTestImages(t);
  Data labels = readTestLables(t);
  Matrix_X x  = buildMatrix_X(t, images);
  Matrix_Y y  = buildMatrix_Y(t, labels);
  free(images.data);
  free(labels.data);

  Matrix_Y y_predict = x.transpose() * theta;
  std::cerr << y - y_predict << "\n";
  size_t counter = 0;
  for (size_t i = 0; i < t; ++i) {
    if (y(maxIndex(y_predict.row(i))) == 1.0f) {
      ++counter;
      std::cerr << i << " ";
    }
  }
  std::cerr << "\n";
  return (float) counter / t;
}

void mnist(int argc, char* argv[]) {
  if (argc < 4) {
    throw std::runtime_error("Too less arguments!");
  }
  const size_t n = atoi(argv[2]);
  const size_t t = atoi(argv[3]);
  Data images    = readTrainImages(n);
  Data labels    = readTrainLables(n);
  Matrix_X x     = buildMatrix_X(n, images);
  Matrix_Y y     = buildMatrix_Y(n, labels);
  free(images.data);
  free(labels.data);

  // solve the system of equations
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXf> cQR(x.transpose());
  Matrix_theta theta = cQR.solve(y);
  float accuracy     = predict(t, theta);

  std::cerr << "Accuracy: " << accuracy << "\n";
}
}  // namespace MNIST
