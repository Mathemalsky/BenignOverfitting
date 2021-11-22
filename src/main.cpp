#include <chrono>
#include <exception>
#include <iostream>
#include <random>

// headers from Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

/*!
 * \brief Samples struct stores all sample datas x_i, y_i
 */
struct Samples {
  Eigen::MatrixXf X;
  Eigen::VectorXf Y;
};

Samples generateSamples(unsigned int n, unsigned int k) {
  Eigen::MatrixXf x(k, n);
  Eigen::VectorXf y(n);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine prng(seed);
  std::normal_distribution<float> distribution(0.0f, 1.0f);
  for (unsigned int j = 0; j < n; ++j) {
    for (unsigned int i = 0; i < k; ++i) {
      x(i, j) = distribution(prng);
    }
    y(j) = distribution(prng);
  }
  return Samples{x, y};
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    throw std::runtime_error("Too less arguments!\n");
  }
  Samples samples = generateSamples(atoi(argv[1]), atoi(argv[2]));
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXf> cQR(samples.X.transpose());
  Eigen::VectorXf theta = cQR.solve(samples.Y);
  std::cout << theta << std::endl;
  return 0;
}
