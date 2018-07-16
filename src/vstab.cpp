#include <iostream>
#include <numeric>

// Boost
#include <boost/program_options.hpp>

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "video.h"
#include "stabilize.h"
#include "crop.h"
#include "fit.h"

boost::program_options::variables_map parse_args(int argc, char *argv[], std::string& out_file) {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
        ("help", "Produces help message")
        ("debug", "Enables debug output")
        ("crop", po::value<float>(), "Crops down to this percent")
        ("file", po::value<std::string>(&out_file)->required(), "The file to process")
        ;
  po::positional_options_description pos;
  pos.add("file", 1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
    if (vm.count("help")) {
      std::cout << "Usage: vstab <FILE> [OPTIONS]" << std::endl;
      std::cout << desc << std::endl;
      return vm;
    }
    po::notify(vm);
  }
  catch (boost::program_options::required_option& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  return vm;
}

void display(const Video& frames) {
  cv::namedWindow("vstab", cv::WINDOW_NORMAL);
  int i = 0;
  char key;
  do {
    const auto& frame = frames[i];
    cv::imshow("vstab", frame);
    key = cv::waitKey();
    if (key == 0x6a) {
      i = (i + 1) % static_cast<int>(frames.size());
    }
    if (key == 0x6B) {
      i--;
      if (i < 0) {
        i = frames.size() - 1;
      }
    }
  } while (key != 27);
  cv::destroyAllWindows();
}

int main(int argc, char *argv[]) {
  std::string file;
  auto options = parse_args(argc, argv, file);
  if (options.count("help")) {
    return 1;
  }
  if (!options.count("file")) {
    return 1;
  }
  const bool debug = static_cast<bool>(options.count("debug"));

  std::cout << "Reading video..." << std::endl;
  Video frames = read_video(file);

  if (options.count("crop")) {
    std::cout << "Cropping video..." << std::endl;
    const float crop_percentage = options["crop"].as<float>();
    crop_to_percentage(frames, crop_percentage);
  }

  std::cout << "Estimating transformations..." << std::endl;
  std::vector<cv::Mat> transforms = stabilize<cv::xfeatures2d::SIFT>(frames, debug);

  std::cout << "Extracting motion..." << std::endl;
  std::vector<cv::Point2f> centers = extract_centers(frames, transforms);

  // The following may go in a loop allowing the change of smoothing parameters.

  std::cout << "Smoothing motion..." << std::endl;
  std::vector<cv::Point2f> centers_smoothed = smooth_motion_parameterless(centers, 80.f);
  add_motion(transforms, centers_smoothed);


  std::cout << "Transforming frames..." << std::endl;
  Video frames_tfed = transform_video(frames, transforms);

  std::cout << "Cropping to common rectangle..." << std::endl;
  std::vector<cv::Rect> rects_cropped = extract_max_cropped_rect(frames, transforms);
  cv::Rect rect_common = std::accumulate(rects_cropped.begin(), rects_cropped.end(), rects_cropped.front(), [](const auto& a, const auto& b) {
      return a & b;
      });
  crop_and_resize(frames_tfed, rect_common);

  // Draw frame centers.
  if (debug) {
    cv::Point2i center_frame = cv::Point2i(frames[0].cols / 2, frames[0].rows / 2);
    for (size_t i = 0; i < frames_tfed.size(); i++) {
      cv::Point2i center_int = static_cast<cv::Point2i>(centers[i]);
      cv::Point2i center_smoothed_int = static_cast<cv::Point2i>(centers_smoothed[i]);
      // Draw trace.
      for (size_t j = i; j < frames_tfed.size(); j++) {
        cv::circle(frames_tfed[j], center_frame + center_int, 2, cv::Scalar(120, 255, 120));
        cv::circle(frames_tfed[j], center_frame + center_smoothed_int, 2, cv::Scalar(120, 120, 255));
      }
    }
  }

  std::cout << "Display..." << std::endl;
  display(frames_tfed);
}
