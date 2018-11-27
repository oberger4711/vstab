#pragma once

#include <vector>
#include <algorithm>

// OpenCV
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "video.h"
#include "track.h"

static inline void printProgress(const size_t frame, const size_t num_frames) {
  std::cout << static_cast<int>(100.0 * frame / num_frames) << " %..." << std::endl;
}

template<typename T>
std::vector<cv::Mat> stabilize(Video& frames, const bool debug) {
  std::vector<cv::Mat> tfs(frames.size());
  std::vector<Track> tracks;
  std::vector<Track> tracks_finished;
  TrackMatcher matcher;
  for (auto& m : tfs) {
    m = cv::Mat::eye(3, 3, CV_64FC1);
  }
  cv::Ptr<T> detector = T::create();
  std::vector<cv::KeyPoint> keypoints_current, keypoints_next;
  cv::Mat descriptors_current, descriptors_next;
  detector->detectAndCompute(frames[0], cv::Mat(), keypoints_next, descriptors_next);
  // Create tracks starting in first frame.
  tracks.reserve(keypoints_next.size());
  for (size_t i = 0; i < keypoints_next.size(); i++) {
    tracks.emplace_back(keypoints_next[i], descriptors_next.row(i));
  }
  // Process following frames.
  for (size_t i = 0; i < frames.size() - 1; i++) {
    if (i % 10 == 0) {
      printProgress(i, frames.size());
    }
    auto& frame_current = frames[i];
    auto& frame_next = frames[i + 1];

    // Apply detector.
    keypoints_current.clear();
    keypoints_current.swap(keypoints_next);
    descriptors_current = descriptors_next.clone();
    detector->detectAndCompute(frame_next, cv::Mat(), keypoints_next, descriptors_next);

    auto& tf_next = tfs[i + 1];
    if (!descriptors_current.empty() && !descriptors_next.empty()) {
        matcher.match(tracks, tracks_finished, keypoints_next, descriptors_next);
        // Debug visualize correspondencies.
        // Estimate transformation.
        /*
        std::vector<unsigned char> mask;
        tf_next = cv::findHomography(pts_next, pts_current, cv::RANSAC, 3, mask);
        if (debug) {
          for (size_t j = 0; j < pts_current.size(); j++) {
            cv::Scalar clr;
            if (mask[j]) {
              clr = cv::Scalar(255, 120, 120);
            }
            else {
              clr = cv::Scalar(120, 120, 255);
            }
            cv::arrowedLine(frame_current, static_cast<cv::Point2i>(pts_current[j]), static_cast<cv::Point2i>(pts_next[j]), clr);
          }
        }
      }
      */
      if (debug) {
        // Viz tracks.
        for (const auto& track : tracks) {
          for (size_t j = 1; j < track.size(); j++) {
            cv::line(frame_current, static_cast<cv::Point2i>(track[j - 1].pt), static_cast<cv::Point2i>(track[j].pt), track.getDebugColor());
          }
        }
        // Viz current keypoints.
        for (const auto& key_point : keypoints_current) {
          cv::circle(frame_current, static_cast<cv::Point2i>(key_point.pt), 1, cv::Scalar(255, 100, 100));
        }
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
  printProgress(frames.size(), frames.size());
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
