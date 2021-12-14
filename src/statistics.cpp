#include "statistics.hpp"

double mean(const std::vector<double>& data) {
  double sum = 0;
  for (const double& el : data) {
    sum += el;
  }
  return sum / data.size();
}

double variance(const std::vector<double>& data) {
  double sum = 0;
  double mu  = mean(data);
  for (const double& el : data) {
    const double a = el - mu;
    sum += (a * a);
  }
  return sum / data.size();
}
