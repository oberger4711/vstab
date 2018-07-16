#include "fit.h"

std::vector<cv::Point2f> smooth_motion_parameterless(const std::vector<cv::Point2f>& centers, const float smoothness) {
  // Initialize parameters as the center points.
  std::vector<double> params(2 * centers.size());
  for (size_t i = 0; i < centers.size(); i++) {
    cv::Point2f center = centers[i];
    params[2 * i] = center.x;
    params[(2 * i) + 1] = center.y;
  }

  google::InitGoogleLogging("vstab");
  ceres::Problem problem;

  // Centered costs
  for (size_t i = 0; i < centers.size(); i++) {
    ceres::CostFunction* cost_centered = new ceres::AutoDiffCostFunction<CenteredCostFunctor, 2, 2>(new CenteredCostFunctor(1.0, centers[i]));
    problem.AddResidualBlock(cost_centered, NULL, &params[2 * i]);
  }

  // Smoothed costs
  for (size_t i = 1; i < centers.size() - 1; i++) {
    ceres::CostFunction* cost_smoothed = new ceres::AutoDiffCostFunction<SmoothedCostFunctor, 2, 2, 2, 2>(new SmoothedCostFunctor(smoothness));
    problem.AddResidualBlock(cost_smoothed, NULL, &params[2 * (i - 1)], &params[2 * i], &params[2 * (i + 1)]);
  }

  // Optimize.
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  // Transform back for returning.
  std::vector<cv::Point2f> centers_smoothed(centers.size());
  for (size_t i = 0; i < centers.size(); i++) {
    centers_smoothed[i] = cv::Point2f(params[2 * i], params[2 * i + 1]);
  }
  return centers_smoothed;
}
