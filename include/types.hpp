#pragma once

#include <cstddef>
#include <vector>

using Byte = unsigned char;

namespace MNIST {
struct Data {
  size_t size;
  Byte* data;
};

struct Data_f {
  size_t size;
  float* data;
};

struct SampleSubset {
  size_t n, k;
  std::vector<size_t> indexes;
};

}  // namespace MNIST
