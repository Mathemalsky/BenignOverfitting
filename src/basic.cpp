#include "basic.hpp"

#include <cstddef>
#include <random>

/*!
 * \brief n_choose_k permutates k random elements into the first k positions of the vector
 * \param n
 * \param k
 * \param indexes
 */
std::vector<size_t> n_choose_k(size_t n, size_t k) {
  std::vector<size_t> indexes(n);
  std::iota(indexes.begin(), indexes.end(), 0);
  srand(time(NULL));
  for (size_t i = 0; i < k; ++i) {
    size_t a = rand() % (n - i);                // select a position in indexes randomly
    std::swap(indexes[i], indexes[n - a - 1]);  // and swap the entry to position i in the vector
  }
  return indexes;
}
