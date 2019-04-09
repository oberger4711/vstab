#pragma once

#include <random>

class BucketSampler {
public:
  using Bucket = std::vector<int>;

  BucketSampler(const cv::Mat& m1, const cv::Mat& m2, const int num_buckets_x=8, const int num_buckets_y=5);
  bool getSubset(cv::Mat& out_ms1, cv::Math out_ms2, const int model_points, const int max_attempts);

private:
  cv::Mat m1;
  cv::Mat m2;
  std::vector<Bucket> buckets;
  std::random_device rd;
  std::mt19937 gen;
  std::discrete_distribution<> dist;
};
