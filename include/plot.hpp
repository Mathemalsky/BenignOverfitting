#pragma once

#include <eigen3/Eigen/Core>

#include "types.hpp"

namespace SIMPLE {
/*!
 * \brief plotCurves
 * \param samples
 * \param computedTheta
 */
void plotCurves(const Samples& samples, const Eigen::VectorXd& computedTheta);

/*!
 * \brief plotTheta
 * \param theta
 */
void plotTheta(const Eigen::VectorXd& theta);
}  // namespace SIMPLE

namespace KERNELESTIMATE {
/*!
 * \brief plotDensity
 * \param data
 * \param h
 */
void plotDensity(const std::vector<double>& data, const double h);
}  // namespace KERNELESTIMATE
