#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <eigen3/Eigen/Core>

#include "constants.hpp"

using Byte = unsigned char;

namespace GENERAL {
using Curve = std::array<std::pair<double, double>, GRID_POINTS>;
}  // namespace GENERAL

namespace MNIST {
/*!
 * \brief Data is a pointer to a byte array an the number of bytes stored there
 */
struct Data {
  size_t size;
  Byte* data;
};

/*!
 * \brief Data_f is a pointer to a float array an the number of floats stored there
 */
struct Data_f {
  size_t size;
  double* data;
};

/*!
 * \brief SampleSubset store a subset of {0,..., n-1} in the fist k entries of indexes
 */
struct SampleSubset {
  size_t n, k;
  std::vector<size_t> indexes;
};

}  // namespace MNIST

namespace SIMPLE {
using Points = std::vector<std::pair<double, double>>;

/*!
 * \brief Samples struct stores all sample datas x_i, y_i, theta_j
 */
struct Samples {
  Eigen::MatrixXd X;
  Eigen::VectorXd Y;
  Eigen::VectorXd Theta;
};
}  // namespace SIMPLE
