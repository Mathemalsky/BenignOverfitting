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
inline const double EPS                     = 0.00001;
}  // namespace MNIST

namespace SIMPLE {
inline const double X_MIN = -1.0;
inline const double X_MAX = 1.0;
}  // namespace SIMPLE

namespace GENERAL {
inline const unsigned int GRID_POINTS = 400;
inline const double REL_PLOT_HEIGHT   = 1.1;
}  // namespace GENERAL

namespace KERNELESTIMATE {
inline const double START        = 0.0;
inline const double END          = 1.0;
inline const double REL_BANDWITH = 9.0;
inline const double EPS          = 1.0 / GENERAL::GRID_POINTS;
}  // namespace KERNELESTIMATE

inline const unsigned int BYTESIZE = 1;
inline const unsigned int MAXBYTE  = 255;
