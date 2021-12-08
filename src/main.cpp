#include "mnist.hpp"
#include "simple.hpp"

int main(int argc, char* argv[]) {
  if (std::strcmp(argv[1], "mnist") == 0) {
    MNIST::mnist(argc, argv);
  }
  else {
    SIMPLE::simple(argc, argv);
  }
  return 0;
}
