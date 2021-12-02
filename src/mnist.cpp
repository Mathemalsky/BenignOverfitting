#include "mnist.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

// headers from Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

#include "constants.hpp"
#include "io.hpp"
#include "types.hpp"

using Matrix_X      = Eigen::Matrix<float, MNIST::PIXELS_PER_IMAGE*(MNIST::DEGREE + 1), Eigen::Dynamic>;
using Matrix_Xt     = Eigen::Matrix<float, Eigen::Dynamic, MNIST::PIXELS_PER_IMAGE*(MNIST::DEGREE + 1)>;
using Matrix_Y      = Eigen::Matrix<float, Eigen::Dynamic, MNIST::POSSIBLE_LABELS>;
using Matrix_theta  = Eigen::Matrix<float, MNIST::PIXELS_PER_IMAGE*(MNIST::DEGREE + 1), MNIST::POSSIBLE_LABELS>;
using VectorLabelsf = Eigen::Matrix<float, 1, MNIST::POSSIBLE_LABELS>;

namespace MNIST {
Matrix_X buildMatrix_X(const size_t n, const Data images) {
  assert((images.size == PIXELS_PER_IMAGE * n) && "Number of bytes read from file does not fit.");

  Matrix_X x(PIXELS_PER_IMAGE * (DEGREE + 1), n);
  for (size_t i = 0; i < PIXELS_PER_IMAGE; ++i) {
    x.row(i) = Eigen::VectorXf::Constant(n, 1);
  }
  for (size_t d = 1; d <= DEGREE; ++d) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t i = 0; i < PIXELS_PER_IMAGE; ++i) {
        x(PIXELS_PER_IMAGE * d + i, j) =
          x(PIXELS_PER_IMAGE * (d - 1) + i, j) * (float) images.data[PIXELS_PER_IMAGE * j + i] / MAXBYTE;
      }
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
  // read test images
  Data images = readTestImages(t);
  Matrix_X x  = buildMatrix_X(t, images);
  free(images.data);

  // read test labels
  Data labels = readTestLables(t);
  Matrix_Y y  = buildMatrix_Y(t, labels);
  free(labels.data);

  Matrix_Y y_predict = x.transpose() * theta;
  size_t counter     = 0;
  for (size_t i = 0; i < t; ++i) {
    if (y(maxIndex(y_predict.row(i))) == 1.0f) {
      ++counter;
    }
  }
  return (float) counter / t;
}

static float effectiveRank_r0(const std::vector<float>& eigenValues) {
  float sum = 0;
  for (const float& lamda : eigenValues) {
    sum += lamda;
  }
  return sum / eigenValues[0];
}

static float effectiveRank_R0(const std::vector<float>& eigenValues) {
  float sum = 0, sumOfSquares = 0;
  for (const float& lambda : eigenValues) {
    sum += lambda;
    sumOfSquares += lambda * lambda;
  }
  return sum * sum / sumOfSquares;
}

void mnist(int argc, char* argv[]) {
  if (argc < 4) {
    throw std::runtime_error("Too less arguments!");
  }
  const size_t n = atoi(argv[2]);
  const size_t t = atoi(argv[3]);

  // read training images
  Data images = readTrainImages(n);
  Matrix_X x  = buildMatrix_X(n, images);
  free(images.data);

  // read training labels
  Data labels = readTrainLables(n);
  Matrix_Y y  = buildMatrix_Y(n, labels);
  free(labels.data);

  // solve the system of equations
  Eigen::CompleteOrthogonalDecomposition<Matrix_Xt> cQR(x.transpose());
  Matrix_theta theta = cQR.solve(y);
  float accuracy     = predict(t, theta);

  // calculate some additional info for output prompt
  Eigen::MatrixXf sigma = x.transpose() * x;        // compute covariance operator sigma
  Eigen::BDCSVD<Eigen::MatrixXf> svd(sigma);        // compute just eigenvalues, no eigenvectors
  const size_t rank = svd.nonzeroSingularValues();  // svd reveals rank of matrix
  std::vector<float> eigenValues(rank);
  auto singularValues = svd.singularValues();
  for (size_t i = 0; i < eigenValues.size(); ++i) {  // the eigenvalues are the squares of the singular values
    eigenValues[i] = singularValues(i) * singularValues(i);
  }

  // display useful information
  std::cerr << "cQR has dims  : " << cQR.rows() << "x" << cQR.cols() << "\n";
  std::cerr << "sigma has rank: " << rank << " and is of size " << sigma.rows() << "x" << sigma.cols() << "\n";
  std::cerr << "r_0 is        : " << effectiveRank_r0(eigenValues) << "\n";
  std::cerr << "R_0 is        : " << effectiveRank_R0(eigenValues) << "\n";
  std::cerr << "Accuracy      : " << accuracy << "\n";
}
}  // namespace MNIST
