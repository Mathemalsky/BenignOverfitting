#pragma once

#include <eigen3/Eigen/Core>

#include "types.hpp"

namespace SIMPLE {
/*!
 * \brief plotCurves plots the supportpoints and the polynomials given by theta and computedTheta as coefficients
 * \param samples contains the original theta and the support points
 * \param computedTheta is the estimate obtained by the benign overfitting
 */
void plotCurves(const Samples& samples, const Eigen::VectorXd& computedTheta);

/*!
 * \brief plotTheta plots the coefficients of theta
 * \param theta is the coefficient vector
 */
void plotTheta(const Eigen::VectorXd& theta);
}  // namespace SIMPLE

namespace KERNELESTIMATE {
/*!
 * \brief plotDensity plots a gaussion kernel estimate of the density for the given data
 * \param data vector of data points
 * \param h bandwidth that is affected by a given data point
 */
void plotDensity(const std::vector<double>& data, const double h);
}  // namespace KERNELESTIMATE
