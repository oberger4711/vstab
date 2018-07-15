#include "video.h"

Video read_video(const std::string& file_name) {
  Video frames;
  cv::VideoCapture cap(file_name);
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open video '" << file_name << "'." << std::endl;
  }
  cv::Mat frame;
  while (cap.grab()) {
    frames.emplace_back();
    auto& frame = frames.back();
    cap.retrieve(frame);
  }
  cap.release();
  return frames;
}
