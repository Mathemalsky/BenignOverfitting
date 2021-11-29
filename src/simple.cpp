#include "simple.hpp"

#include <chrono>
#include <iostream>
#include <random>

Samples generateSamples(unsigned int n, unsigned int k) {
  // allocate memory
  Eigen::MatrixXf x(k, n);
  Eigen::VectorXf y(n);
  Eigen::VectorXf theta(k);

  // set up random number generation
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine prng(seed);
  std::normal_distribution<float> distribution(0.0f, 1.0f);
  std::normal_distribution<float> distribution2(0.0f, 0.1f);

  // generate random theta
  for (unsigned int i = 0; i < k; ++i) {
    theta[i] = distribution(prng);
  }

  // generate random x and y using theta
  for (unsigned int j = 0; j < n; ++j) {
    y(j) = distribution2(prng);
    for (unsigned int i = 0; i < k; ++i) {
      x(i, j) = distribution(prng);
      y(j) += x(i, j) * theta[i];
    }
  }
  return Samples{x, y, theta};
}

void simple(int argc, char* argv[]) {
  if (argc < 3) {
    throw std::runtime_error("Too less arguments!\n");
  }
  const unsigned int k = atoi(argv[2]);
  Samples samples      = generateSamples(atoi(argv[1]), k);  // get the random samples

  // solve the system of equations
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXf> cQR(samples.X.transpose());
  Eigen::VectorXf theta = cQR.solve(samples.Y);

  // format for output
  Eigen::MatrixXf output(k, 2);
  output.col(0) = samples.Theta;
  output.col(1) = theta;
  std::cout << output << std::endl;
  std::cout << "X^T * theta - y =\n" << samples.X.transpose() * theta - samples.Y << "\n";
  std::cout << "relative loss: " << (theta - samples.Theta).norm() / (float) k << "\n";
}
