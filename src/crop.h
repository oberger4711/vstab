#pragma once

#include <vector>
#include <algorithm>

#include "video.h"

// OpenCV
#include "opencv2/calib3d.hpp"

/**
 * Extracts an axis-aligned bounding box with maximal size inside the transformed frame, that contains no black border pixels.
 * @param frames The video frames.
 * @param transforms The transformation matrices for each frame.
 * @return The maximal cropped rectangle for each frame.
 */
std::vector<cv::Rect> extract_max_cropped_rect(const Video& frames, const std::vector<cv::Mat>& transforms);

/**
 * Extracts the center of the axis-aligned bounding box with maximal size inside the transformed frame, that contains no black border pixels.
 * @param frames The video frames.
 * @param transforms The transformation matrices for each frame.
 * @return The center points for each frame.
 */
std::vector<cv::Point2f> extract_centers(const Video& frames, const std::vector<cv::Mat>& transforms);

void crop_to_percentage(Video& frames, const float percentage);

void crop_and_resize(Video& frames, const cv::Rect& rect_common);
