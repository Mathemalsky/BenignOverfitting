#include "io.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

#include "constants.hpp"
#include "types.hpp"

namespace MNIST {

static Data read(const char* filename, size_t count) {
  FILE* file = fopen(filename, "rb");
  if (!file) {
    throw std::runtime_error((std::string) "File <" + filename + "> not found");
  }
  Byte* data = (Byte*) malloc(count);
  Byte* bin  = (Byte*) malloc(HEADER_SIZE);

// disable gcc warning -Wunsused-result
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"

  fread(bin, BYTESIZE, HEADER_SIZE, file);
  free(bin);
  fread(data, BYTESIZE, count, file);

// enable gcc warning -Wunused-result
#pragma GCC diagnostic pop
  return Data{count, data};
}

Data readTrainImages(size_t imagecount) {
  assert(imagecount < TRAINING_SET_SIZE && "Number of training samples to read should not exeed data size!");
  return read(TRAIN_IMAGE_FILE, imagecount * PIXELS_PER_IMAGE);
}

Data readTrainLables(size_t labelcount) {
  assert(labelcount < TRAINING_SET_SIZE && "Number of training samples to read should not exeed data size!");
  return read(TRAIN_LABEL_FILE, labelcount);
}

Data readTestImages(size_t imagecount) {
  assert(imagecount < TEST_SET_SIZE && "Number of test samples to read should not exeed data size!");
  return read(TEST_IMAGE_FILE, imagecount * PIXELS_PER_IMAGE);
}

Data readTestLables(size_t labelcount) {
  assert(labelcount < TEST_SET_SIZE && "Number of test samples to read should not exeed data size!");
  return read(TEST_LABEL_FILE, labelcount);
}
}  // namespace MNIST
