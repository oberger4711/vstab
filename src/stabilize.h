#pragma once

#include <vector>
#include <algorithm>

// OpenCV
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "video.h"

template<typename T>
std::vector<cv::Mat> stabilize(Video& frames, const bool debug) {
  const auto eye = cv::Mat::eye(3, 3, CV_64FC1);
  std::vector<cv::Mat> tfs(frames.size(), eye);
  cv::Ptr<T> detector = T::create();

  std::vector<cv::KeyPoint> keypoints_current, keypoints_next;
  cv::Mat descriptors_current, descriptors_next;
  detector->detectAndCompute(frames[0], cv::Mat(), keypoints_next, descriptors_next);
  for (size_t i = 0; i < frames.size() - 1; i++) {
    auto& frame_current = frames[i];
    auto& frame_next = frames[i + 1];

    // Apply detector.
    keypoints_current.clear();
    keypoints_current.swap(keypoints_next);
    descriptors_current = descriptors_next;
    detector->detectAndCompute(frame_next, cv::Mat(), keypoints_next, descriptors_next);

    auto& tf_next = tfs[i + 1];
    if (!descriptors_current.empty() && !descriptors_next.empty()) {
      // Find keypoint matches.
      cv::FlannBasedMatcher matcher;
      std::vector<std::vector<cv::DMatch>> matches_all;
      matcher.knnMatch(descriptors_current, descriptors_next, matches_all, 2);

      // Filter matches.
      // Keep only good matches.
      std::vector<cv::DMatch> matches_good;
      for (const auto& neighbours : matches_all) {
        if (neighbours[0].distance < 0.75 * neighbours[1].distance) {
          matches_good.push_back(neighbours[0]);
        }
      }
      if (!matches_good.empty()) {
        // Keep only close matches. I. e. closer than the median distance.
        std::vector<float> distances(matches_good.size());
        for (size_t i = 0; i < matches_good.size(); i++) {
          const auto& pt_current = keypoints_current[matches_good[i].queryIdx].pt;
          const auto& pt_next = keypoints_next[matches_good[i].trainIdx].pt;
          distances[i] = cv::norm(pt_next - pt_current);
        }
        std::sort(distances.begin(), distances.end());
        const float median = distances[distances.size() / 2];
        for (auto it = matches_good.begin(); it != matches_good.end();) {
          const auto& pt_current = keypoints_current[it->queryIdx].pt;
          const auto& pt_next = keypoints_next[it->trainIdx].pt;
          if (cv::norm(pt_next - pt_current) > median) {
            it = matches_good.erase(it);
          }
          else {
            ++it;
          }
        }

        // Extract corresponding keypoints.
        std::vector<cv::Point2f> pts_current, pts_next;
        for (const auto& match : matches_good) {
          pts_current.push_back(keypoints_current[match.queryIdx].pt);
          pts_next.push_back(keypoints_next[match.trainIdx].pt);
        }

        // Debug visualize correspondencies.
        if (debug) {
          for (size_t j = 0; j < pts_current.size(); j++) {
            cv::arrowedLine(frame_current, static_cast<cv::Point2i>(pts_current[j]), static_cast<cv::Point2i>(pts_next[j]), cv::Scalar(255, 120, 120));
          }
        }
        // Estimate transformation.
        tf_next = cv::findHomography(pts_next, pts_current, cv::RANSAC);
      }
    }
    else {
      // No keypoints available for estimation.
      std::cerr << "Warning: No keypoints in current or next frame." << std::endl;
    }

    if (tf_next.empty()) {
      std::cerr << "Warning: Empty homography for frame " << i << "." << std::endl;
    }
    tf_next = tfs[i] * tf_next; // Accumulate transforms.
  }
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
