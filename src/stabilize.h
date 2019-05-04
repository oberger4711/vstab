#pragma once

#include <vector>
#include <algorithm>

// OpenCV
#include "opencv2/video.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "video.h"
#include "registration.h"

struct StabilizationParameters {
  float min_key_frame_overlap = 0.3;
};

std::vector<cv::DMatch> matchKeyPoints(const cv::Mat& descriptors_reference, const cv::Mat& descriptors_next);

template<typename T>
std::vector<cv::Mat> stabilize(Video& frames, const bool debug) {
  const StabilizationParameters params;
  std::vector<cv::Mat> tfs(frames.size());
  for (auto& m : tfs) {
    m = cv::Mat::eye(3, 3, CV_64FC1);
  }
  cv::Ptr<T> detector = T::create();
  std::vector<cv::KeyPoint> key_points_key_frame, key_points_current, key_points_next;
  cv::Mat descriptors_key_frame, descriptors_current, descriptors_next;
  detector->detectAndCompute(frames[0], cv::Mat(), key_points_next, descriptors_next);
  key_points_key_frame = key_points_next;
  descriptors_key_frame = descriptors_next;

  std::cout << "0 %" << std::endl;
  size_t i_key_frame = 0;
  auto& tf_key_frame = tfs[0];
  for (size_t i = 0; i < frames.size() - 1; i++) {
    // Progress output.
    const size_t percent = static_cast<size_t>((static_cast<float>(i) / frames.size()) * 100);
    std::cout << "\e[1A" << percent << " %   " << std::endl;

    // Apply detector on next frame.
    auto& frame_next = frames[i + 1];
    key_points_current.clear();
    key_points_current.swap(key_points_next);
    descriptors_current = descriptors_next.clone();
    detector->detectAndCompute(frame_next, cv::Mat(), key_points_next, descriptors_next);

    auto& tf_current = tfs[i];
    auto& tf_next = tfs[i + 1];

    bool matched_with_good_overlap = false;
    cv::Mat tf_next_estimated = cv::Mat::eye(3, 3, CV_64FC1);
    std::vector<cv::Point2f> good_key_points_key_frame, good_key_points_next;
    std::vector<int> inlier_mask;
    // Try to match against key frame first but create new key frame if overlap is too low.
    if (descriptors_next.empty()) {
      // No key points available for estimation.
      std::cerr << "Warning: No key points in next frame. Skipping frame." << std::endl << std::endl;
      continue;
    }
    // Find key point matches.
    std::vector<cv::DMatch> matches_good = matchKeyPoints(descriptors_key_frame, descriptors_next);

    // Extract corresponding key points.
    for (const auto& match : matches_good) {
      good_key_points_key_frame.push_back(key_points_key_frame[match.queryIdx].pt);
      good_key_points_next.push_back(key_points_next[match.trainIdx].pt);
    }

    // Estimate transform.
    inlier_mask.clear();
    //tf_next_estimated = cv::findHomography(good_key_points_next, good_key_points_key_frame, cv::RANSAC, 3.0, inlier_mask);
    estimateRigidTransform_extended(good_key_points_next, good_key_points_key_frame, false, SamplingMethod::RANSAC, inlier_mask)
      .copyTo(tf_next_estimated(cv::Range(0, 2), cv::Range::all()));

    // New key frame if too few associations.
    const size_t num_inliers = static_cast<size_t>(std::accumulate(inlier_mask.cbegin(), inlier_mask.cend(), 0));
    const float overlap = static_cast<float>(num_inliers) / key_points_next.size();
    //const size_t num_inliers = matches_good.size();
    // TODO: Other possible key frame policies:
    // - IOU between rectangle key frame and transformed next frame < threshold.
    // - IOU between set of key points transformed into each other frame and then lying inside the other rectangle.
    if (i_key_frame != i &&
        !descriptors_current.empty() &&
        (matches_good.empty() ||
         overlap < params.min_key_frame_overlap)) {
      // New key frame due to small overlap.
      std::cout << "New key frame. Overlap was " << overlap << "." << std::endl << std::endl;
      // Update key frame.
      i_key_frame = i;
      tf_key_frame = tf_current;
      key_points_key_frame = key_points_current;
      descriptors_key_frame = descriptors_current;

      // Match and estimate again.
      // TODO: Refactor this. Problem: Depends on many variables and visualization stuff. Maybe use OOP (Create class for match / estimation with attributes).
      matches_good = matchKeyPoints(descriptors_key_frame, descriptors_next);
      good_key_points_key_frame.clear();
      good_key_points_next.clear();
      inlier_mask.clear();
      for (const auto& match : matches_good) {
        good_key_points_key_frame.push_back(key_points_key_frame[match.queryIdx].pt);
        good_key_points_next.push_back(key_points_next[match.trainIdx].pt);
      }
      //tf_next_estimated = cv::findHomography(good_key_points_next, good_key_points_key_frame, cv::RANSAC, 3.0, inlier_mask);
      estimateRigidTransform_extended(good_key_points_next, good_key_points_key_frame, false, SamplingMethod::RANSAC, inlier_mask)
        .copyTo(tf_next_estimated(cv::Range(0, 2), cv::Range::all()));
    }

    if (matches_good.empty()) {
      // No matches available for estimation.
      std::cerr << "Warning: No matches to estimate transformation." << std::endl << std::endl;
    }

    // Debug visualize correspondencies.
    if (debug) {
      assert(inlier_mask.size() == good_key_points_key_frame.size());
      std::array<cv::Scalar, 2> colors = {cv::Scalar(100, 100, 255), cv::Scalar(255, 120, 120)};
      for (size_t j = 0; j < good_key_points_key_frame.size(); j++) {
        auto& color = colors[inlier_mask[j]];
        cv::arrowedLine(frame_next,
            static_cast<cv::Point2i>(good_key_points_key_frame[j]),
            static_cast<cv::Point2i>(good_key_points_next[j]), color);
      }
    }

    if (tf_next_estimated.empty()) {
      std::cerr << "Warning: Empty estimated transform for frame " << i << "." << std::endl;
      std::cout << std::endl;
    }
    // Accumulate transforms.
    tf_next = tf_key_frame * tf_next_estimated;
  }
  std::cout << "\e[1A" << "100 %   " << std::endl;
  return tfs;
}

void add_motion(std::vector<cv::Mat>& in_out_transforms, const std::vector<cv::Point2f>& centers);

/**
 * Applies the transformations on the frames
 * @param frames The video frames.
 * @param transforms The transformation matrices for each frame.
 * @return Transformed copy of the frames.
 */
Video transform_video(Video& frames, const std::vector<cv::Mat>& transforms);
