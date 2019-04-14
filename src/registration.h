#pragma once

#include <vector>

// OpenCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>

#include "bucket_ransac.h"
#include "lm_solver.h"

template<typename T> inline int compressElems( T* ptr, const uchar* mask, int mstep, int count )
{
    int i, j;
    for( i = j = 0; i < count; i++ )
        if( mask[i*mstep] )
        {
            if( i > j )
                ptr[j] = ptr[i];
            j++;
        }
    return j;
}

enum class RegistrationMethod {
  ALL = 0,
  RANSAC = cv::RANSAC,
  LMEDS = cv::LMEDS,
  BUCKET_RANSAC
};

cv::Mat find_homography_extended(cv::InputArray pts_current, cv::InputArray pts_next,
    const RegistrationMethod method = RegistrationMethod::ALL);

cv::Mat find_homography_extended_ransac(cv::InputArray srcPoints, cv::InputArray dstPoints,
    double ransacReprojThreshold = 3,
    cv::OutputArray mask=cv::noArray(), const int maxIters = 2000,
    const double confidence = 0.995);
