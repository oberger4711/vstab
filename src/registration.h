#pragma once

#include <vector>

// OpenCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

enum class RegistrationMethod {
  ALL = 0,
  RANSAC = cv::RANSAC,
  BUCKET_RANSAC
};

cv::Mat estimateRigidTransform_extended(cv::InputArray src1, cv::InputArray src2, bool fullAffine);
cv::Mat estimateRigidTransform_extended(cv::InputArray src1, cv::InputArray src2, bool fullAffine,
                                   int ransacMaxIters, double ransacGoodRatio,
                                   const int ransacSize0);
