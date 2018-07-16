#include "crop.h"

std::vector<cv::Rect> extract_max_cropped_rect(const Video& frames, const std::vector<cv::Mat>& transforms) {
  std::vector<cv::Rect> cropped_rects(frames.size());
  for (size_t i = 0; i < frames.size(); i++) {
    const auto& frame = frames[i];
    const auto& tf = transforms[i];
    const double width = static_cast<double>(frame.cols);
    const double height = static_cast<double>(frame.rows);

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
    const double top = max;
    // Bottom
    cv::minMaxLoc(corners_tfed_ch[1].colRange(0, 2), &min, &max);
    const double bottom = min;
    // Right
    cv::minMaxLoc(corners_tfed_ch[0].colRange(1, 3), &min, &max);
    const double right = min;
    // Left
    cv::minMaxLoc(corners_tfed_ch[0].colRange(3, 5), &min, &max);
    const double left = max;

    cropped_rects[i] = cv::Rect(left, top, (right - left), (bottom - top));
  }
  return cropped_rects;
}

std::vector<cv::Point2f> extract_centers(const Video& frames, const std::vector<cv::Mat>& transforms) {
  std::vector<cv::Point2f> centers(frames.size());
  std::vector<cv::Rect> cropped_rects = extract_max_cropped_rect(frames, transforms);
  for (size_t i = 0; i < cropped_rects.size(); i++) {
    const auto& rect = cropped_rects[i];
    centers[i] = (rect.tl() + rect.br()) / 2.0;
  }
  return centers;
}
