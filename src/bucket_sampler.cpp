#include "bucket_sampler.h"

BucketSampler::BucketSampler(const cv::Mat& m1, const cv::Mat& m2, const int num_buckets_x, const int num_buckets_y) :
  m1(m1), m2(m2), rd(), gen(rd) {
  
  const float bucket_width = 
  // Sort points into buckets.
  for (int i = 0; i < 
}
