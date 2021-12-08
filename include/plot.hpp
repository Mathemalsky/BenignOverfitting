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
}  // namespace SIMPLE
