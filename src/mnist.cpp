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

namespace MNIST {
Eigen::MatrixXd buildMatrix_X(const size_t n, const Data images) {
  assert((images.size == PIXELS_PER_IMAGE * n) && "Number of bytes read from file does not fit.");

  Eigen::MatrixXd x(PIXELS_PER_IMAGE * (DEGREE + 1), n);
  for (size_t i = 0; i < PIXELS_PER_IMAGE; ++i) {
    x.row(i) = Eigen::VectorXd::Constant(n, 1);
  }
  for (size_t d = 1; d <= DEGREE; ++d) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t i = 0; i < PIXELS_PER_IMAGE; ++i) {
        x(PIXELS_PER_IMAGE * d + i, j) =
          x(PIXELS_PER_IMAGE * (d - 1) + i, j) * (double) images.data[PIXELS_PER_IMAGE * j + i] / MAXBYTE;
      }
    }
  }
  return x;
}

Eigen::MatrixXd buildMatrix_Y(const size_t n, const Data labels) {
  assert(labels.size == n && "Number of labels read and n are missmatching!");

  Eigen::MatrixXd y(n, POSSIBLE_LABELS);
  for (size_t i = 0; i < n; ++i) {
    for (unsigned char j = 0; j < POSSIBLE_LABELS; ++j) {
      y(i, j) = (j == labels.data[i]) ? 1.0f : 0.0f;
    }
  }
  return y;
}

static size_t maxIndex(const Eigen::VectorXd vec) {
  assert(vec.size() != 0 && "Vector should be non empty!");
  size_t index = 0;
  double max   = vec(0);
  for (size_t i = 1; i < (size_t) vec.size(); ++i) {
    if (vec(i) > max) {
      max   = vec(i);
      index = i;
    }
  }
  return index;
}

double predict(const size_t t, const Eigen::MatrixXd theta) {
  // read test images
  Data images       = readTestImages(t);
  Eigen::MatrixXd x = buildMatrix_X(t, images);
  free(images.data);

  // read test labels
  Data labels       = readTestLables(t);
  Eigen::MatrixXd y = buildMatrix_Y(t, labels);
  free(labels.data);

  Eigen::MatrixXd y_predict = x.transpose() * theta;
  size_t counter            = 0;
  for (size_t i = 0; i < t; ++i) {
    if (y(maxIndex(y_predict.row(i))) == 1.0f) {
      ++counter;
    }
  }
  return (double) counter / t;
}

static double effectiveRank_r0(const std::vector<double>& eigenValues) {
  assert(eigenValues.size() > 0 && "length of vector shoul be nonzero!");
  double sum = 0;
  for (const double& lamda : eigenValues) {
    sum += lamda;
  }
  return sum / eigenValues[0];
}

static double effectiveRank_R0(const std::vector<double>& eigenValues) {
  assert(eigenValues.size() > 0 && "length of vector shoul be nonzero!");
  double sum = 0, sumOfSquares = 0;
  for (const double& lambda : eigenValues) {
    sum += lambda;
    sumOfSquares += lambda * lambda;
  }
  return sum * sum / sumOfSquares;
}

std::array<unsigned int, POSSIBLE_LABELS> countLabels(const Data& data) {
  std::array<unsigned int, POSSIBLE_LABELS> counter;
  for (unsigned int i = 0; i < POSSIBLE_LABELS; ++i) {
    counter[i] = 0;
  }
  for (unsigned int i = 0; i < data.size; ++i) {
    ++counter[data.data[i]];
  }
  return counter;
}

void displayLabelCounter(const std::array<unsigned int, POSSIBLE_LABELS>& labelCounter) {
  std::cout << "Labels appeared in the training set in the following distribution:\n\n";
  for (unsigned int i = 0; i < POSSIBLE_LABELS; ++i) {
    printf("     %u |", i);
  }
  printf("\n");
  for (unsigned int i = 0; i < POSSIBLE_LABELS; ++i) {
    printf("-------|");
  }
  printf("\n");
  for (unsigned int i = 0; i < POSSIBLE_LABELS; ++i) {
    printf(" %5s |", (std::to_string(labelCounter[i])).c_str());
  }
  printf("\n\n");
}

void mnist(int argc, char* argv[]) {
  if (argc < 5) {
    throw std::runtime_error("Too less arguments!");
  }
  const size_t n  = atoi(argv[2]);
  const size_t t  = atoi(argv[3]);
  const double mu = atof(argv[4]);

  // read training images
  Data images       = readTrainImages(n);
  Eigen::MatrixXd x = buildMatrix_X(n, images);
  free(images.data);

  // read training labels
  Data labels                                            = readTrainLables(n);
  Eigen::MatrixXd y                                      = buildMatrix_Y(n, labels);
  std::array<unsigned int, POSSIBLE_LABELS> labelCounter = countLabels(labels);
  free(labels.data);

  double accuracy;
  if (mu <= EPS) {
    // solve the system of equations
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cQR(x.transpose());
    Eigen::MatrixXd theta = cQR.solve(y);
    accuracy              = predict(t, theta);
  }
  else {
    Eigen::MatrixXd A = x * x.transpose();
    for (unsigned int d = 0; d <= DEGREE; ++d) {
      for (unsigned int i = 0; i < PIXELS_PER_IMAGE; ++i) {
        const unsigned int index = d * PIXELS_PER_IMAGE + i;
        A(index, index) += mu * d;
      }
    }
    Eigen::MatrixXd theta = A.ldlt().solve(x * y);
    accuracy              = predict(t, theta);
  }

  // calculate some additional info for output prompt
  Eigen::MatrixXd sigma = x.transpose() * x;        // compute covariance operator sigma
  Eigen::BDCSVD<Eigen::MatrixXd> svd(sigma);        // compute just eigenvalues, no eigenvectors
  const size_t rank = svd.nonzeroSingularValues();  // svd reveals rank of matrix
  std::vector<double> eigenValues(rank);
  Eigen::VectorXd singularValues = svd.singularValues();
  for (size_t i = 0; i < eigenValues.size(); ++i) {  // the eigenvalues are the squares of the singular values
    eigenValues[i] = singularValues(i) * singularValues(i);
  }

  if (argc == 5) {
    // display useful information
    displayLabelCounter(labelCounter);
    std::cerr << "sigma has rank: " << rank << " and is of size " << sigma.rows() << "x" << sigma.cols() << "\n";
    std::cerr << "r_0 is        : " << effectiveRank_r0(eigenValues) << "\n";
    std::cerr << "R_0 is        : " << effectiveRank_R0(eigenValues) << "\n";
    std::cerr << "Accuracy      : " << accuracy << "\n";
  }
  else if (std::strcmp(argv[5], "-d") == 0) {
    writeAccuracy(accuracy);
  }
}
}  // namespace MNIST
