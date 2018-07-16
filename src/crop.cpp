#include "crop.h"

std::vector<cv::Rect> extract_max_cropped_rect(const Video& frames, const std::vector<cv::Mat>& transforms) {
  std::vector<cv::Rect> rects_cropped(frames.size());
  for (size_t i = 0; i < frames.size(); i++) {
    const auto& frame = frames[i];
    const auto& tf = transforms[i];
    const double width = static_cast<double>(frame.cols);
    const double height = static_cast<double>(frame.rows);
    const double aspect_ratio = width / height;

    // First point is also inserted as last point to make colRange work easily.
    double data[10] = {
      0, height,
      width, height,
      width, 0,
      0, 0,
      0, height};
    cv::Mat corners(1, 5, CV_64FC2, &data[0]);
    cv::Mat corners_tfed(1, 5, CV_64FC2);
    cv::perspectiveTransform(corners, corners_tfed, tf);
    cv::Mat corners_tfed_ch[2];
    cv::split(corners_tfed, corners_tfed_ch);

    // Determine cropped rectangle.
    double min, max;
    // Top
    cv::minMaxLoc(corners_tfed_ch[1].colRange(2, 4), &min, &max);
    const double top_content = max;
    // Bottom
    cv::minMaxLoc(corners_tfed_ch[1].colRange(0, 2), &min, &max);
    const double bottom_content = min;
    // Right
    cv::minMaxLoc(corners_tfed_ch[0].colRange(1, 3), &min, &max);
    const double right_content = min;
    // Left
    cv::minMaxLoc(corners_tfed_ch[0].colRange(3, 5), &min, &max);
    const double left_content = max;

    // Respect aspect ratio.
    const double width_content = right_content - left_content;
    const double height_content = bottom_content - top_content;
    const double center_x = left_content + (width_content / 2.0);
    const double center_y = top_content + (height_content / 2.0);
    double width_cropped, height_cropped;
    if (height_content * aspect_ratio > width_content) {
      // Width is limiting.
      width_cropped = width_content;
      height_cropped = width_content / aspect_ratio;
    }
    else {
      // Height is limiting.
      height_cropped = height_content;
      width_cropped = height_content * aspect_ratio;
    }
    rects_cropped[i] = cv::Rect(center_x - (width_cropped / 2.0), center_y - (height_cropped / 2.0), width_cropped, height_cropped);
  }
  return rects_cropped;
}

std::vector<cv::Point2f> extract_centers(const Video& frames, const std::vector<cv::Mat>& transforms) {
  std::vector<cv::Point2f> centers(frames.size());
  std::vector<cv::Rect> rects_cropped = extract_max_cropped_rect(frames, transforms);
  for (size_t i = 0; i < rects_cropped.size(); i++) {
    const auto& rect = rects_cropped[i];
    centers[i] = rect.tl();
  }
  return centers;
}

void crop_to_percentage(Video& frames, const float percentage) {
  const auto& frame = frames[0];
  const int width = frame.cols;
  const int height = frame.rows;
  const int center_x = width / 2;
  const int center_y = height / 2;
  const int width_new = static_cast<int>(width * percentage);
  const int height_new = static_cast<int>(height * percentage);
  cv::Rect roi(center_x - width_new / 2, center_y - height_new / 2, width_new, height_new);

  for (auto& frame : frames) {
    frame = frame(roi);
  }
}

void crop_and_resize(Video& frames, const cv::Rect& rect_common) {
  for (auto& frame : frames) {
    frame = frame(rect_common);
  }
}
