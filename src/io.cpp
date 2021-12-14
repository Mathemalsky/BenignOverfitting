#include "io.hpp"

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "basic.hpp"
#include "constants.hpp"

namespace MNIST {

static Data read(const char* filename, const size_t size, const SampleSubset& subset) {
  FILE* file = fopen(filename, "rb");
  if (!file) {
    throw std::runtime_error((std::string) "File <" + filename + "> not found");
  }
  Byte* rawdata = (Byte*) malloc(size * subset.n);
  Byte* data    = (Byte*) malloc(size * subset.k);
  Byte* bin     = (Byte*) malloc(HEADER_SIZE);

// disable gcc warning -Wunsused-result
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"

  fread(bin, BYTESIZE, HEADER_SIZE, file);
  free(bin);
  fread(rawdata, BYTESIZE, size * subset.n, file);

// enable gcc warning -Wunused-result
#pragma GCC diagnostic pop

  for (size_t i = 0; i < subset.k; ++i) {
    std::memcpy(data + i * size, rawdata + size * subset.indexes[i], size);
  }
  free(rawdata);
  return Data{size * subset.k, data};
}

Data readTrainImages(size_t imagecount) {
  assert(imagecount <= TRAINING_SET_SIZE && "Number of training samples to read should not exeed data size!");
  SampleSubset subset{TRAINING_SET_SIZE, imagecount, n_choose_k(TRAINING_SET_SIZE, imagecount)};
  return read(TRAIN_IMAGE_FILE, PIXELS_PER_IMAGE, subset);
}

Data readTrainLables(size_t labelcount) {
  assert(labelcount <= TRAINING_SET_SIZE && "Number of training samples to read should not exeed data size!");
  SampleSubset subset{TRAINING_SET_SIZE, labelcount, n_choose_k(TRAINING_SET_SIZE, labelcount)};
  return read(TRAIN_LABEL_FILE, BYTESIZE, subset);
}

Data readTestImages(size_t imagecount) {
  assert(imagecount <= TEST_SET_SIZE && "Number of test samples to read should not exeed data size!");
  SampleSubset subset{TRAINING_SET_SIZE, imagecount, n_choose_k(TEST_SET_SIZE, imagecount)};
  return read(TEST_IMAGE_FILE, PIXELS_PER_IMAGE, subset);
}

Data readTestLables(size_t labelcount) {
  assert(labelcount <= TEST_SET_SIZE && "Number of test samples to read should not exeed data size!");
  SampleSubset subset{TRAINING_SET_SIZE, labelcount, n_choose_k(TRAINING_SET_SIZE, labelcount)};
  return read(TEST_LABEL_FILE, BYTESIZE, subset);
}

void writeAccuracy(const double accuracy) {
  std::ofstream file(OUTPUTFILE, std::ios_base::app);
  if (!file) {
    throw std::runtime_error((std::string) "Couldn't create file <" + OUTPUTFILE + ">.");
  }
  file << accuracy << "\n";
}
}  // namespace MNIST

std::vector<double> readAccuracy(const char* filename) {
  std::ifstream file(filename, std::ios::in);
  if (!file) {
    throw std::runtime_error((std::string) "File <" + filename + "> not found.");
  }
  std::vector<double> data;
  double number;
  while (file >> number) {
    data.push_back(number);
  }
  return data;
}
