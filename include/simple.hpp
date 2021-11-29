#pragma once

// headers from Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

/*!
 * \brief Samples struct stores all sample datas x_i, y_i, theta_j
 */
struct Samples {
  Eigen::MatrixXf X;
  Eigen::VectorXf Y;
  Eigen::VectorXf Theta;
};

/*!
 * \brief generateSamples
 * \param n
 * \param k
 * \return
 */
Samples generateSamples(unsigned int n, unsigned int k);

/*!
 * \brief simple
 * \param argc
 * \param argv
 */
void simple(int argc, char* argv[]);
