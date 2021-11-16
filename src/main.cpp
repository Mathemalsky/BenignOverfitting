#include <random>
#include <eigen3/Eigen/Core>

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

  return Samples{x, y};
}

int main(int argc, char* argv[]) {
  return 0;
}
