#include "simple.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <eigen3/Eigen/QR>

#include "gnuplot-iostream.h"

#include "constants.hpp"
#include "plot.hpp"
#include "types.hpp"

namespace SIMPLE {
Samples generateSamples(const unsigned int n, const unsigned int k, const double mu) {
  // allocate memory
  Eigen::MatrixXd x(k, n + k - 1);
  Eigen::VectorXd y(n + k - 1);
  Eigen::VectorXd theta(k);
  std::vector<double> supportPoints(n);

  // set up random number generation
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine prng(seed);
  std::uniform_real_distribution<double> distribution_support(X_MIN, X_MAX);
  std::uniform_real_distribution<double> distribution_coefficients(-5.0, 5.0);
  std::normal_distribution<double> distribution_noise(0, 0.25);

  // generate random theta
  for (unsigned int i = 0; i < k; ++i) {
    theta[i] = distribution_coefficients(prng);
  }

  // generate support points
  for (unsigned int j = 0; j < n; ++j) {
    supportPoints[j] = distribution_support(prng);
  }
  std::sort(supportPoints.begin(), supportPoints.end());

  // generate random x and y using theta
  for (unsigned int j = 0; j < n; ++j) {
    y(j)    = distribution_noise(prng) + theta[0];
    x(0, j) = 1;
    for (unsigned int i = 1; i < k; ++i) {
      x(i, j) = x(i - 1, j) * supportPoints[j];
      y(j) += x(i, j) * theta[i];
    }
  }
  // add regularization part of the matrix
  for (unsigned int j = n; j < n + k - 1; ++j) {
    y(j) = 0;
    for (unsigned int i = 0; i < k; ++i) {
      x(i, j) = (j - n + 1 == i) ? std::sqrt(mu * (j - n + 1)) : 0.0f;
    }
  }
  return Samples{x, y, theta};
}

void simple(int argc, char* argv[]) {
  if (argc < 3) {
    throw std::runtime_error("Too less arguments!\n");
  }
  const unsigned int k = atoi(argv[2]);
  Samples samples      = generateSamples(atoi(argv[1]), k, atof(argv[3]));  // get the random samples

  // solve the system of equations
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cQR(samples.X.transpose());
  Eigen::VectorXd theta = cQR.solve(samples.Y);

  plotCurves(samples, theta);
  plotTheta(theta);
}
}  // namespace SIMPLE
