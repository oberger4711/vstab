#include <iostream>

// Boost
#include <boost/program_options.hpp>

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"

#include "video.h"
#include "stabilize.h"

boost::program_options::variables_map parse_args(int argc, char *argv[], std::string& out_file) {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
        ("help", "Produces help message")
        ("debug", "Enables debug output")
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

  std::cout << "Estimating transformations..." << std::endl;
  std::vector<cv::Mat> transforms = stabilize<cv::xfeatures2d::SIFT>(frames, debug);

  std::cout << "Display..." << std::endl;
  display(frames);
}
