#pragma once

#include <vector>
#include <cstdlib>
#include <iostream>

// OpenCV
#include "opencv2/features2d.hpp"

class Track {
public:
  Track(const cv::KeyPoint& keypoint_init, const cv::Mat& descriptor_init);
  void associate(const cv::KeyPoint& keypoint_associated, const cv::Mat& descriptor_associated);
  cv::Mat getDescriptor() const;
  cv::KeyPoint getCurrentKeypoint() const;
  cv::KeyPoint operator[](const size_t i) const;
  size_t size() const;
  cv::Scalar getDebugColor() const;

private:
  std::vector<cv::KeyPoint> keypoints_history;
  cv::Mat descriptor_latest;
  cv::Scalar color_debug;
};

class TrackMatcher {
public:
  void match(std::vector<Track>& tracks, std::vector<Track>& tracks_finished, const std::vector<cv::KeyPoint>& keypoints_next, const cv::Mat& descriptors_next);
};
