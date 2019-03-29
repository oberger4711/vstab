#pragma once

#include <opencv2/core/core.hpp>

class LMSolver : public cv::Algorithm
{
public:
    class Callback
    {
    public:
        virtual ~Callback() {}
        virtual bool compute(cv::InputArray param, cv::OutputArray err, cv::OutputArray J) const = 0;
    };

    virtual void setCallback(const cv::Ptr<LMSolver::Callback>& cb) = 0;
    virtual int run(cv::InputOutputArray _param0) const = 0;
};

cv::Ptr<LMSolver> createLMSolver(const cv::Ptr<LMSolver::Callback>& cb, int maxIters);
