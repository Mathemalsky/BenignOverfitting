#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <eigen3/Eigen/Core>

#include "constants.hpp"

using Byte = unsigned char;

namespace MNIST {
struct Data {
  size_t size;
  Byte* data;
};

struct Data_f {
  size_t size;
  float* data;
};

struct SampleSubset {
  size_t n, k;
  std::vector<size_t> indexes;
};

}  // namespace MNIST

namespace SIMPLE {

using Curve  = std::array<std::pair<float, float>, GRID_POINTS>;
using Points = std::vector<std::pair<float, float>>;

/*!
 * \brief Samples struct stores all sample datas x_i, y_i, theta_j
 */
struct Samples {
  Eigen::MatrixXf X;
  Eigen::VectorXf Y;
  Eigen::VectorXf Theta;
};
}  // namespace SIMPLE
