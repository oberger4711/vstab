#include "track.h"

Track::Track(const cv::KeyPoint& keypoint_init, const cv::Mat& descriptor_init) : descriptor_latest(descriptor_init.clone()), color_debug(std::rand() % 255, std::rand() % 255, std::rand() % 255) {
  keypoints_history.push_back(keypoint_init);
}

void Track::associate(const cv::KeyPoint& keypoint_associated, const cv::Mat& descriptor_associated) {
  keypoints_history.push_back(keypoint_associated);
  descriptor_associated.copyTo(descriptor_latest);
}

cv::Mat Track::getDescriptor() const {
  return descriptor_latest;
}

cv::KeyPoint Track::getCurrentKeypoint() const {
  return keypoints_history.back();
}

cv::KeyPoint Track::operator[](const size_t i) const {
  if (i > keypoints_history.size()) {
    std::cerr << "ERROR: Track point out of bounds. Requested index " << i << " is greater or equal to size " << keypoints_history.size() << "." << std::endl;
  }
  return keypoints_history[i];
}

size_t Track::size() const {
  return keypoints_history.size();
}

cv::Scalar Track::getDebugColor() const {
  return color_debug;
}

void TrackMatcher::match(std::vector<Track>& tracks, std::vector<Track>& tracks_finished, const std::vector<cv::KeyPoint>& keypoints_next, const cv::Mat& descriptors_next) {
  cv::Mat descriptors_tracks(tracks.size(), 128, CV_32F);
  for (size_t i = 0; i < tracks.size(); i++) {
    auto desc_row = descriptors_tracks.row(i);
    tracks[i].getDescriptor().copyTo(desc_row);
  }
  // Find keypoint matches.
  cv::FlannBasedMatcher matcher;
  std::vector<std::vector<cv::DMatch>> matches_all;
  matcher.knnMatch(descriptors_tracks, descriptors_next, matches_all, 2);

  // Filter matches.
  // Keep only good matches.
  std::vector<cv::DMatch> matches_good;
  for (const auto& neighbours : matches_all) {
    if (neighbours[0].distance < 0.75 * neighbours[1].distance) {
      matches_good.push_back(neighbours[0]);
    }
  }
  if (!matches_good.empty()) {
    // Keep only close matches. I. e. closer than the median distance.
    std::vector<float> distances(matches_good.size());
    for (size_t i = 0; i < matches_good.size(); i++) {
      const auto& pt_current = tracks[matches_good[i].queryIdx].getCurrentKeypoint().pt;
      const auto& pt_next = keypoints_next[matches_good[i].trainIdx].pt;
      distances[i] = cv::norm(pt_next - pt_current);
    }
    std::sort(distances.begin(), distances.end());
    const float median = 1.1 * distances[distances.size() / 2];
    for (auto it = matches_good.begin(); it != matches_good.end();) {
      const auto& pt_current = tracks[it->queryIdx].getCurrentKeypoint().pt;
      const auto& pt_next = keypoints_next[it->trainIdx].pt;
      if (cv::norm(pt_next - pt_current) > median) {
        it = matches_good.erase(it);
      }
      else {
        ++it;
      }
    }

    // Update tracks and detect finished tracks.
    std::vector<bool> finished(tracks.size(), true);
    for (const auto& match : matches_good) {
      const auto j = match.queryIdx;
      tracks[j].associate(keypoints_next[match.trainIdx], descriptors_next.row(match.trainIdx));
      finished[j] = false;
    }
    if (!finished.empty()) {
      for (unsigned long j = static_cast<unsigned long>(finished.size()) - 1; j > 0; j--) {
        if (finished[j]) {
          tracks_finished.push_back(tracks[j]);
          tracks.erase(tracks.begin() + j);
        }
      }
    }
  }
}
