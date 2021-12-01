#pragma once

namespace MNIST {
inline const char* TRAIN_IMAGE_FILE         = "mnist/train-images.idx3-ubyte";
inline const char* TRAIN_LABEL_FILE         = "mnist/train-labels.idx1-ubyte";
inline const char* TEST_IMAGE_FILE          = "mnist/t10k-images.idx3-ubyte";
inline const char* TEST_LABEL_FILE          = "mnist/t10k-labels.idx1-ubyte";
inline const unsigned int HEADER_SIZE       = 16;
inline const unsigned int TRAINING_SET_SIZE = 60000;
inline const unsigned int TEST_SET_SIZE     = 10000;
inline const unsigned int PIXELS_PER_IMAGE  = 28 * 28;
inline const unsigned char POSSIBLE_LABELS  = 10;
inline const unsigned int DEGREE            = 1;
}  // namespace MNIST

inline const unsigned int BYTESIZE = 1;
inline const unsigned int MAXBYTE  = 255;
