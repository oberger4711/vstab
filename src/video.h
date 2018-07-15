#pragma once

#include <vector>
#include <string>
#include <iostream>

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using Video = std::vector<cv::Mat>;

Video read_video(const std::string& file_name);
