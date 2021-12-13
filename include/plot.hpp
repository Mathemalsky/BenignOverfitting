#pragma once

#include <eigen3/Eigen/Core>

#include "types.hpp"

namespace SIMPLE {
/*!
 * \brief plotCurves
 * \param samples
 * \param computedTheta
 */
void plotCurves(const Samples& samples, const Eigen::VectorXf& computedTheta);

/*!
 * \brief plotTheta
 * \param theta
 */
void plotTheta(const Eigen::VectorXf& theta);
}  // namespace SIMPLE

namespace KERNELESTIMATE {
/*!
 * \brief plotDensity
 * \param data
 * \param h
 */
void plotDensity(const std::vector<float>& data, const float h);
}  // namespace KERNELESTIMATE
