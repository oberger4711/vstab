#pragma once

#include <opencv2/core/core.hpp>

class BucketRANSACPointSetRegistrator : public cv::Algorithm
{
  public:
    class Callback
    {
      public:
        virtual ~Callback() = default;
        virtual int runKernel(cv::InputArray m1, cv::InputArray m2, cv::OutputArray model) const = 0;
        virtual void computeError(cv::InputArray m1, cv::InputArray m2, cv::InputArray model, cv::OutputArray err) const = 0;
        virtual bool checkSubset(cv::InputArray, cv::InputArray, int) const { return true; }
    };

    BucketRANSACPointSetRegistrator(const cv::Ptr<BucketRANSACPointSetRegistrator::Callback>& _cb=cv::Ptr<BucketRANSACPointSetRegistrator::Callback>(),
                          int _modelPoints=0, double _threshold=0, double _confidence=0.99, int _maxIters=1000);
    int findInliers(const cv::Mat& m1, const cv::Mat& m2, const cv::Mat& model, cv::Mat& err, cv::Mat& mask, double thresh) const;
    bool getSubset( const cv::Mat& m1, const cv::Mat& m2, cv::Mat& ms1, cv::Mat& ms2, cv::RNG& rng, int maxAttempts=1000 ) const;
    virtual void setCallback(const cv::Ptr<BucketRANSACPointSetRegistrator::Callback>& cb);
    virtual bool run(cv::InputArray m1, cv::InputArray m2, cv::OutputArray model, cv::OutputArray mask) const;
  private:
    cv::Ptr<BucketRANSACPointSetRegistrator::Callback> cb;
    int modelPoints;
    double threshold;
    double confidence;
    int maxIters;
};

cv::Ptr<BucketRANSACPointSetRegistrator> createBucketRANSACPointSetRegistrator(const cv::Ptr<BucketRANSACPointSetRegistrator::Callback>& cb,
    int modelPoints, double threshold,
    double confidence=0.99, int maxIters=1000);
