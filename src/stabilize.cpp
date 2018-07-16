#include "stabilize.h"

Video transform_video(Video& frames, std::vector<cv::Mat> transforms) {
  Video frames_tfed(frames.size());
  for (size_t i = 0; i < frames_tfed.size(); i++) {
    const auto& frame = frames[i];
    auto& frame_tfed = frames_tfed[i];
    const auto& tf = transforms[i];
    cv::warpPerspective(frame, frame_tfed, tf, frame_tfed.size());
  }
  return frames_tfed;
}
