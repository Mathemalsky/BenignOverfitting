#pragma once

#include "types.hpp"

namespace SIMPLE {

/*!
 * \brief generateSamples
 * \param n
 * \param k
 * \param mu
 * \return
 */
Samples generateSamples(const unsigned int n, const unsigned int k, const double mu);

/*!
 * \brief simple
 * \param argc
 * \param argv
 */
void simple(int argc, char* argv[]);
}  // namespace SIMPLE
