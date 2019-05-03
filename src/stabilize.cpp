#include "stabilize.h"

std::vector<cv::DMatch> matchKeyPoints(const cv::Mat& descriptors_reference, const cv::Mat& descriptors_next) {
  cv::FlannBasedMatcher matcher;
  std::vector<std::vector<cv::DMatch>> matches_all;
  matcher.knnMatch(descriptors_reference, descriptors_next, matches_all, 2);

  // Filter matches.
  // Keep only good matches.
  std::vector<cv::DMatch> matches_good;
  for (const auto& neighbours : matches_all) {
    if (neighbours[0].distance < 0.75 * neighbours[1].distance) {
      matches_good.push_back(neighbours[0]);
    }
  }
  return matches_good;
}

void add_motion(std::vector<cv::Mat>& in_out_transforms, const std::vector<cv::Point2f>& centers) {
  for (size_t i = 0; i < in_out_transforms.size(); i++) {
    auto& tf = in_out_transforms[i];
    const auto t = centers[i];
    tf.at<double>(0, 2) -= t.x;
    tf.at<double>(1, 2) -= t.y;
  }
}

Video transform_video(Video& frames, const std::vector<cv::Mat>& transforms) {
  Video frames_tfed(frames.size());
  for (size_t i = 0; i < frames_tfed.size(); i++) {
    const auto& frame = frames[i];
    auto& frame_tfed = frames_tfed[i];
    const auto& tf = transforms[i];
    cv::warpPerspective(frame, frame_tfed, tf, frame_tfed.size());
  }
  return frames_tfed;
}
