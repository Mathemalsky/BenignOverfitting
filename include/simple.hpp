#pragma once

#include "types.hpp"

namespace SIMPLE {

/*!
 * \brief generateSamples
 * \param n number of training data
 * \param k degrees of freedom
 * \param mu weight for regularization penalty
 * \return sample containing X, theta and y
 */
Samples generateSamples(const unsigned int n, const unsigned int k, const double mu);

/*!
 * \brief simple fitting a polynomial to supportpoints
 * \param argc
 * \param argv
 */
void simple(int argc, char* argv[]);
}  // namespace SIMPLE
