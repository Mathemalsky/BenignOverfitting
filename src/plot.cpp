#include "plot.hpp"

#include <array>

#include "gnuplot-iostream.h"

#include "constants.hpp"

using namespace GENERAL;

static void initCurve(Curve& curve, const double x_min, const double x_max) {
  const double stepsize = (x_max - x_min) / GRID_POINTS;
  for (unsigned int i = 0; i < GRID_POINTS; ++i) {
    curve[i].first  = x_min + i * stepsize;
    curve[i].second = 0.0f;
  }
}

namespace SIMPLE {
void evalPolynomial(Curve& curve, const Eigen::VectorXf& coeff) {
  const unsigned int k = coeff.size();
  for (unsigned int j = 0; j < k; ++j) {
    for (unsigned int i = 0; i < GRID_POINTS; ++i) {
      curve[i].second *= curve[i].first;
      curve[i].second += coeff(k - 1 - j);
    }
  }
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
  const unsigned int n = samples.X.cols() - samples.X.rows() + 1;
  const double x_min   = samples.X(1, 0);
  const double x_max   = samples.X(1, n - 1);

  Curve pointsOrigTheta;
  initCurve(pointsOrigTheta, x_min, x_max);
  evalPolynomial(pointsOrigTheta, samples.Theta);

  Curve pointsComputedTheta;
  initCurve(pointsComputedTheta, x_min, x_max);
  evalPolynomial(pointsComputedTheta, computedTheta);

  Points supportPoints = extractPoints(samples);

  std::vector<std::pair<double, double>> xy_pts_B;
  for (double alpha = 0; alpha < 1; alpha += 1.0 / 24.0) {
    double theta = alpha * 2.0 * 3.14159;
    xy_pts_B.push_back(std::make_pair(cos(theta), sin(theta)));
  }

  Gnuplot gp;
  gp << "set terminal png size 700,400\n";
  gp << "set output 'theta_plot.png'\n";
  gp << "set xlabel 'x'\n";
  gp << "set ylabel 'y' rotate by 0\n";
  gp << "plot" << gp.file1d(pointsOrigTheta) << "with lines title 'theta'," << gp.file1d(pointsComputedTheta)
     << "with lines title 'regression theta'," << gp.file1d(supportPoints) << "with points title 'support points'\n";
  gp << "set output\n";
}

void plotTheta(const Eigen::VectorXf& theta) {
  const unsigned int k = theta.size();
  Points points(k);
  for (unsigned int i = 0; i < k; ++i) {
    points[i].first  = i;
    points[i].second = theta(i);
  }

  Gnuplot gp;
  gp << "set terminal png size 700,400\n";
  gp << "set output 'theta_coefficients.png'\n";
  gp << "set xlabel 'degree'\n";
  gp << "set ylabel 'coefficient'\n";
  gp << "plot" << gp.file1d(points) << "with points title 'coefficients of theta',\n";
  gp << "set output\n";
}
}  // namespace SIMPLE

namespace KERNELESTIMATE {
static double gauss(const double x, const double sigma) {
  static const double inv_sqrt_2pi = 0.3989422804014327f;
  const double xDivSigma           = x / sigma;
  return inv_sqrt_2pi * std::exp(-0.5 * xDivSigma * xDivSigma) / sigma;
}

Curve kernelEstimate(const std::vector<double>& data, const double h) {
  const double stepsize = (END - START) / GRID_POINTS;
  const unsigned int n  = data.size();
  Curve curve;
  initCurve(curve, START, END);
  for (unsigned int i = 0; i < n; ++i) {
    const double start          = data[i] - h / 2;
    const double end            = data[i] + h / 2;
    const int startindex        = std::ceil((start - START) / stepsize);
    const unsigned int endindex = std::floor((end - START) / stepsize);
    for (int j = startindex; j <= 0; ++j) {
      curve[0].second += gauss(START + j * stepsize - data[i], h / REL_BANDWITH);
    }
    for (unsigned int j = std::max(0, startindex); j <= endindex && j < GRID_POINTS; ++j) {
      curve[j].second += gauss(curve[j].first - data[i], h / REL_BANDWITH);
    }
    /*
    for (unsigned int j = GRID_POINTS; j <= endindex; ++j) {
      curve[GRID_POINTS - 1].second += gauss(START + j * stepsize - data[i], h / REL_BANDWITH);
    }
    */
  }
  for (unsigned int j = 0; j < GRID_POINTS; ++j) {
    curve[j].second /= (double) n;
  }
  return curve;
}

void plotDensity(const std::vector<double>& data, const double h) {
  Curve curve = kernelEstimate(data, h);
  Gnuplot gp;
  gp << "set terminal png size 1400,800\n";
  gp << "set output 'density.png'\n";
  gp << "set yrange [0.0 : 5.0]\n";
  gp << "set xlabel 'accuracy'\n";
  gp << "plot" << gp.file1d(curve) << "with lines title 'density of accuracy',\n";
  gp << "set output\n";
}
}  // namespace KERNELESTIMATE
