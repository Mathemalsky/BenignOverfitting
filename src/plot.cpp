#include "plot.hpp"

#include <array>

#include "gnuplot-iostream.h"

#include "constants.hpp"

namespace SIMPLE {

Curve evalPolynomial(const std::array<float, GRID_POINTS>& grid, const Eigen::VectorXf& coeff) {
  Curve points;
  // initialize points
  for (unsigned int i = 0; i < GRID_POINTS; ++i) {
    points[i].first  = grid[i];
    points[i].second = 0.0f;
  }
  const unsigned int k = coeff.size();
  for (unsigned int j = 0; j < k; ++j) {
    for (unsigned int i = 0; i < GRID_POINTS; ++i) {
      points[i].second *= points[i].first;
      points[i].second += coeff(k - 1 - j);
    }
  }
  return points;
}

Points extractPoints(const Samples& samples) {
  const unsigned int n = samples.X.cols() - samples.X.rows() + 1;
  Points points(n);
  for (unsigned int i = 0; i < n; ++i) {
    points[i].first  = samples.X(1, i);
    points[i].second = samples.Y(i);
  }
  return points;
}

void plotCurves(const Samples& samples, const Eigen::VectorXf& computedTheta) {
  Gnuplot gp;
  const unsigned int n = samples.X.cols() - samples.X.rows() + 1;
  const float x_min    = samples.X(1, 0);
  const float x_max    = samples.X(1, n - 1);

  const float stepsize = (x_max - x_min) / GRID_POINTS;
  std::array<float, GRID_POINTS> grid;
  for (unsigned int i = 0; i < GRID_POINTS; ++i) {
    grid[i] = x_min + i * stepsize;
  }

  Curve pointsOrigTheta     = evalPolynomial(grid, samples.Theta);
  Curve pointscomputedTheta = evalPolynomial(grid, computedTheta);
  Points supportPoints      = extractPoints(samples);

  std::vector<std::pair<double, double>> xy_pts_B;
  for (double alpha = 0; alpha < 1; alpha += 1.0 / 24.0) {
    double theta = alpha * 2.0 * 3.14159;
    xy_pts_B.push_back(std::make_pair(cos(theta), sin(theta)));
  }

  // gp << "set xrange [" << x_min << ":" << x_max << "]\n";
  gp << "plot" << gp.file1d(pointsOrigTheta) << "with lines title 'theta'," << gp.file1d(pointscomputedTheta)
     << "with lines title 'regression theta'," << gp.file1d(supportPoints) << "with points title 'support points'"
     << "\n";
}

void plotTheta(const Eigen::VectorXf& theta) {
  const unsigned int k = theta.size();
  Points points(k);
  for (unsigned int i = 0; i < k; ++i) {
    points[i].first  = i;
    points[i].second = theta(i);
  }

  Gnuplot gp;
  gp << "plot" << gp.file1d(points) << "with points title 'coefficients of theta',\n";
}
}  // namespace SIMPLE
