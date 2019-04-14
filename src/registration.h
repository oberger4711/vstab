#pragma once

#include <vector>

// OpenCV
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc/imgproc.hpp>

enum class SamplingMethod {
  ALL,
  RANSAC,
  BUCKET_RANSAC
};

cv::Mat estimateRigidTransform_extended(cv::InputArray src1, cv::InputArray src2, bool fullAffine,
                                        const SamplingMethod method, std::vector<int>& inlier_mask);

cv::Mat estimateRigidTransform_extended(cv::InputArray src1, cv::InputArray src2, const bool fullAffine,
                                        const SamplingMethod method, std::vector<int>& inlier_mask,
                                        int ransacMaxIters, double ransacGoodRatio);
