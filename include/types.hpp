#pragma once

#include <cstddef>

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

}  // namespace MNIST
