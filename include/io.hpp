#pragma once

#include <vector>
#include "types.hpp"

namespace MNIST {
Data readTrainImages(size_t imagecount);
Data readTrainLables(size_t labelcount);
Data readTestImages(size_t imagecount);
Data readTestLables(size_t labelcount);
void writeAccuracy(const float accuracy);
}  // namespace MNIST

std::vector<float> readAccuracy(const char* filename);
