#pragma once

#include <vector>
#include "types.hpp"

namespace MNIST {
Data readTrainImages(size_t imagecount);
Data readTrainLables(size_t labelcount);
Data readTestImages(size_t imagecount);
Data readTestLables(size_t labelcount);
void writeAccuracy(const double accuracy);
}  // namespace MNIST

std::vector<double> readAccuracy(const char* filename);
