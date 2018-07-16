#pragma once

#include <array>
#include <vector>
#include <cassert>

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>

// Ceres
#include <ceres/ceres.h>
#include <glog/logging.h>

std::vector<cv::Point2f> smooth_motion_parameterless(const std::vector<cv::Point2f>& centers, const float smoothness);

/**
 * Defines difference of the smoothed and the actual center as costs.
 */
struct CenteredCostFunctor {
  CenteredCostFunctor(const float weight, const cv::Point2f center) : weight(weight), center_x(center.x), center_y(center.y) {
  }

  template <typename T> bool operator()(const T* const params, T* residual) const {
    T t_x(params[0]), t_y(params[1]);
    residual[0] = T(weight) * (t_x - T(center_x));
    residual[1] = T(weight) * (t_y - T(center_y));
    return true;
  }

private:
  float weight;
  double center_x;
  double center_y;
};

/**
 * Defines difference of two consecutive steps between smoothed centers as costs.
 */
struct SmoothedCostFunctor {
  SmoothedCostFunctor(const float weight) : weight(weight) {
  }

  template <typename T> bool operator()(const T* params_prev, const T* const params, const T* const params_next, T* residual) const {
    T t_prev_x(params_prev[0]), t_prev_y(params_prev[1]);
    T t_cur_x(params[0]), t_cur_y(params[1]);
    T t_next_x(params_next[0]), t_next_y(params_next[1]);

    T s_prev_x = t_cur_x - t_prev_x;
    T s_prev_y = t_cur_y - t_prev_y;
    T s_next_x = t_next_x - t_cur_x;
    T s_next_y = t_next_y - t_cur_y;

    residual[0] = T(weight) * (s_next_x - s_prev_x);
    residual[1] = T(weight) * (s_next_y - s_prev_y);
    return true;
  }

private:
  float weight;
};
