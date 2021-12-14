#include "io.hpp"
#include "mnist.hpp"
#include "plot.hpp"
#include "simple.hpp"

int main(int argc, char* argv[]) {
  if (std::strcmp(argv[1], "mnist") == 0) {
    MNIST::mnist(argc, argv);
  }
  else if (std::strcmp(argv[1], "density") == 0) {
    const char* filename = argv[2];
    const double h       = atof(argv[3]);
    KERNELESTIMATE::plotDensity(readAccuracy(filename), h);
  }
  else {
    SIMPLE::simple(argc, argv);
  }
  return 0;
}
