#include "bucket_ransac.h"

static inline int RANSACUpdateNumIters( double p, double ep, int modelPoints, int maxIters )
{
    if( modelPoints <= 0 )
        CV_Error(cv::Error::StsOutOfRange, "the number of model points should be positive");

    p = MAX(p, 0.);
    p = MIN(p, 1.);
    ep = MAX(ep, 0.);
    ep = MIN(ep, 1.);

    // avoid inf's & nan's
    double num = MAX(1. - p, DBL_MIN);
    double denom = 1. - std::pow(1. - ep, modelPoints);
    if( denom < DBL_MIN )
        return 0;

    num = std::log(num);
    denom = std::log(denom);

    return denom >= 0 || -num >= maxIters*(-denom) ? maxIters : cvRound(num/denom);
}

BucketRANSACPointSetRegistrator::BucketRANSACPointSetRegistrator(const cv::Ptr<BucketRANSACPointSetRegistrator::Callback>& _cb,
    int _modelPoints, double _threshold, double _confidence, int _maxIters) :
    cb(_cb),
    modelPoints(_modelPoints),
    threshold(_threshold),
    confidence(_confidence),
    maxIters(_maxIters) {
}

int BucketRANSACPointSetRegistrator::findInliers(const cv::Mat& m1, const cv::Mat& m2, const cv::Mat& model, cv::Mat& err, cv::Mat& mask, double thresh) const
{
    cb->computeError( m1, m2, model, err );
    mask.create(err.size(), CV_8U);

    CV_Assert( err.isContinuous() && err.type() == CV_32F && mask.isContinuous() && mask.type() == CV_8U);
    const float* errptr = err.ptr<float>();
    uchar* maskptr = mask.ptr<uchar>();
    float t = (float)(thresh*thresh);
    int i, n = (int)err.total(), nz = 0;
    for( i = 0; i < n; i++ )
    {
        int f = errptr[i] <= t;
        maskptr[i] = (uchar)f;
        nz += f;
    }
    return nz;
}

bool BucketRANSACPointSetRegistrator::getSubset(const cv::Mat& m1, const cv::Mat& m2,
                cv::Mat& ms1, cv::Mat& ms2, cv::RNG& rng,
                int maxAttempts) const
{
    cv::AutoBuffer<int> _idx(modelPoints);
    int* idx = _idx.data();
    int i = 0, j, k, iters = 0;
    int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
    int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
    int esz1 = (int)m1.elemSize1()*d1, esz2 = (int)m2.elemSize1()*d2;
    int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
    const int *m1ptr = m1.ptr<int>(), *m2ptr = m2.ptr<int>();

    ms1.create(modelPoints, 1, CV_MAKETYPE(m1.depth(), d1));
    ms2.create(modelPoints, 1, CV_MAKETYPE(m2.depth(), d2));

    int *ms1ptr = ms1.ptr<int>(), *ms2ptr = ms2.ptr<int>();

    CV_Assert( count >= modelPoints && count == count2 );
    CV_Assert( (esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0 );
    esz1 /= sizeof(int);
    esz2 /= sizeof(int);

    for(; iters < maxAttempts; iters++)
    {
        for( i = 0; i < modelPoints && iters < maxAttempts; )
        {
            int idx_i = 0;
            for(;;)
            {
                idx_i = idx[i] = rng.uniform(0, count);
                for( j = 0; j < i; j++ )
                    if( idx_i == idx[j] )
                        break;
                if( j == i )
                    break;
            }
            for( k = 0; k < esz1; k++ )
                ms1ptr[i*esz1 + k] = m1ptr[idx_i*esz1 + k];
            for( k = 0; k < esz2; k++ )
                ms2ptr[i*esz2 + k] = m2ptr[idx_i*esz2 + k];
            i++;
        }
        if( i == modelPoints && !cb->checkSubset(ms1, ms2, i) )
            continue;
        break;
    }

    return i == modelPoints && iters < maxAttempts;
}

bool BucketRANSACPointSetRegistrator::run(cv::InputArray _m1, cv::InputArray _m2, cv::OutputArray _model, cv::OutputArray _mask) const
{
    bool result = false;
    cv::Mat m1 = _m1.getMat(), m2 = _m2.getMat();
    cv::Mat err, mask, model, bestModel, ms1, ms2;

    int iter, niters = MAX(maxIters, 1);
    int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
    int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
    int count = m1.checkVector(d1), count2 = m2.checkVector(d2), maxGoodCount = 0;

    cv::RNG rng((uint64)-1);

    CV_Assert( cb );
    CV_Assert( confidence > 0 && confidence < 1 );

    CV_Assert( count >= 0 && count2 == count );
    if( count < modelPoints )
        return false;

    cv::Mat bestMask0, bestMask;

    if( _mask.needed() )
    {
        _mask.create(count, 1, CV_8U, -1, true);
        bestMask0 = bestMask = _mask.getMat();
        CV_Assert( (bestMask.cols == 1 || bestMask.rows == 1) && (int)bestMask.total() == count );
    }
    else
    {
        bestMask.create(count, 1, CV_8U);
        bestMask0 = bestMask;
    }

    if( count == modelPoints )
    {
        if( cb->runKernel(m1, m2, bestModel) <= 0 )
            return false;
        bestModel.copyTo(_model);
        bestMask.setTo(cv::Scalar::all(1));
        return true;
    }

    for( iter = 0; iter < niters; iter++ )
    {
        int i, nmodels;
        if( count > modelPoints )
        {
            bool found = getSubset( m1, m2, ms1, ms2, rng, 10000 );
            if( !found )
            {
                if( iter == 0 )
                    return false;
                break;
            }
        }

        nmodels = cb->runKernel( ms1, ms2, model );
        if( nmodels <= 0 )
            continue;
        CV_Assert( model.rows % nmodels == 0 );
        cv::Size modelSize(model.cols, model.rows/nmodels);

        for( i = 0; i < nmodels; i++ )
        {
          cv::Mat model_i = model.rowRange( i*modelSize.height, (i+1)*modelSize.height );
            int goodCount = findInliers( m1, m2, model_i, err, mask, threshold );

            if( goodCount > MAX(maxGoodCount, modelPoints-1) )
            {
                std::swap(mask, bestMask);
                model_i.copyTo(bestModel);
                maxGoodCount = goodCount;
                niters = RANSACUpdateNumIters( confidence, (double)(count - goodCount)/count, modelPoints, niters );
            }
        }
    }

    if( maxGoodCount > 0 )
    {
        if( bestMask.data != bestMask0.data )
        {
            if( bestMask.size() == bestMask0.size() )
                bestMask.copyTo(bestMask0);
            else
                transpose(bestMask, bestMask0);
        }
        bestModel.copyTo(_model);
        result = true;
    }
    else
        _model.release();

    return result;
}

void BucketRANSACPointSetRegistrator::setCallback(const cv::Ptr<BucketRANSACPointSetRegistrator::Callback>& _cb) {
  cb = _cb;
}

cv::Ptr<BucketRANSACPointSetRegistrator> createBucketRANSACPointSetRegistrator(const cv::Ptr<BucketRANSACPointSetRegistrator::Callback>& cb,
    int modelPoints, double threshold,
    double confidence, int maxIters) {
  return cv::Ptr<BucketRANSACPointSetRegistrator>(
      new BucketRANSACPointSetRegistrator(cb, modelPoints, threshold, confidence, maxIters));
}
