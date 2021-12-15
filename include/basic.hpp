#pragma once

#include <vector>

/*!
 * \brief n_choose_k selects randomly k number from {0, ... , n-1}
 * \param n
 * \param k
 * \return vector of integers, beginning with the k choosen
 */
std::vector<size_t> n_choose_k(size_t n, size_t k);
