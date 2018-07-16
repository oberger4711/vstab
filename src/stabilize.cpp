#include "stabilize.h"

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
