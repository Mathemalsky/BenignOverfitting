#pragma once

#include <vector>
#include "types.hpp"

namespace MNIST {
/*!
 * \brief readTrainImages
 * \param imagecount
 * \return
 */
Data readTrainImages(size_t imagecount);

/*!
 * \brief readTrainLables
 * \param labelcount
 * \return
 */
Data readTrainLables(size_t labelcount);

/*!
 * \brief readTestImages
 * \param imagecount
 * \return
 */
Data readTestImages(size_t imagecount);

/*!
 * \brief readTestLables
 * \param labelcount
 * \return
 */
Data readTestLables(size_t labelcount);

/*!
 * \brief writeAccuracy
 * \param accuracy
 */
void writeAccuracy(const double accuracy);
}  // namespace MNIST

/*!
 * \brief readAccuracy
 * \param filename
 * \return
 */
std::vector<double> readAccuracy(const char* filename);
