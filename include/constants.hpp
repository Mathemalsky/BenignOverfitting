#pragma once

namespace MNIST {
inline const char* TRAIN_IMAGE_FILE         = "mnist/train-images.idx3-ubyte";
inline const char* TRAIN_LABEL_FILE         = "mnist/train-labels.idx1-ubyte";
inline const char* TEST_IMAGE_FILE          = "mnist/t10k-images.idx3-ubyte";
inline const char* TEST_LABEL_FILE          = "mnist/t10k-labels.idx1-ubyte";
inline const char* OUTPUTFILE               = "accuracy.dat";
inline const unsigned int HEADER_SIZE       = 16;
inline const unsigned int TRAINING_SET_SIZE = 60000;
inline const unsigned int TEST_SET_SIZE     = 10000;
inline const unsigned int PIXELS_PER_IMAGE  = 28 * 28;
inline const unsigned char POSSIBLE_LABELS  = 10;
inline const unsigned int DEGREE            = 3;
}  // namespace MNIST

namespace SIMPLE {
inline const float X_MIN = -1.0f;
inline const float X_MAX = 1.0f;
}  // namespace SIMPLE

namespace GENERAL {
inline const unsigned int GRID_POINTS = 400;
}

namespace KERNELESTIMATE {
inline const float START        = 0.0f;
inline const float END          = 1.0f;
inline const float REL_BANDWITH = 6.0f;
}  // namespace KERNELESTIMATE

inline const unsigned int BYTESIZE = 1;
inline const unsigned int MAXBYTE  = 255;
