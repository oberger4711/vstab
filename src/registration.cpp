#include "registration.h"

#include "opencv2/opencv_modules.hpp"

cv::Mat find_homography_extended(cv::InputArray pts_current, cv::InputArray pts_next,
    const RegistrationMethod method) {
  if (method == RegistrationMethod::BUCKET_RANSAC) {
    // Use own implementation of RANSAC improved by bucket selection.
    // See "A robust technique for matching two uncalibrated images through the recovery of the unknown epipolar geometry".
    return find_homography_extended_ransac(pts_current, pts_next);
  }
  else {
    // Use standard CV implementation.
    return cv::findHomography(pts_current, pts_next, static_cast<int>(method));
  }
}

static inline bool haveCollinearPoints(const cv::Mat& m, int count)
{
    int j, k, i = count-1;
    const cv::Point2f* ptr = m.ptr<cv::Point2f>();

    // check that the i-th selected point does not belong
    // to a line connecting some previously selected points
    // also checks that points are not too close to each other
    for (j = 0; j < i; j++)
    {
        double dx1 = ptr[j].x - ptr[i].x;
        double dy1 = ptr[j].y - ptr[i].y;
        for (k = 0; k < j; k++)
        {
            double dx2 = ptr[k].x - ptr[i].x;
            double dy2 = ptr[k].y - ptr[i].y;
            if (fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
                return true;
        }
    }
    return false;
}

class HomographyEstimatorCallback : public BucketRANSACPointSetRegistrator::Callback
{
public:
    bool checkSubset(cv::InputArray _ms1, cv::InputArray _ms2, int count) const CV_OVERRIDE
    {
        cv::Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();
        if (haveCollinearPoints(ms1, count) || haveCollinearPoints(ms2, count))
            return false;

        // We check whether the minimal set of points for the homography estimation
        // are geometrically consistent. We check if every 3 correspondences sets
        // fulfills the constraint.
        //
        // The usefullness of this constraint is explained in the paper:
        //
        // "Speeding-up homography estimation in mobile devices"
        // Journal of Real-Time Image Processing. 2013. DOI: 10.1007/s11554-012-0314-1
        // Pablo Marquez-Neila, Javier Lopez-Alberca, Jose M. Buenaposada, Luis Baumela
        if (count == 4)
        {
            static const int tt[][3] = {{0, 1, 2}, {1, 2, 3}, {0, 2, 3}, {0, 1, 3}};
            const cv::Point2f* src = ms1.ptr<cv::Point2f>();
            const cv::Point2f* dst = ms2.ptr<cv::Point2f>();
            int negative = 0;

            for (int i = 0; i < 4; i++)
            {
                const int* t = tt[i];
                cv::Matx33d A(src[t[0]].x, src[t[0]].y, 1., src[t[1]].x, src[t[1]].y, 1., src[t[2]].x, src[t[2]].y, 1.);
                cv::Matx33d B(dst[t[0]].x, dst[t[0]].y, 1., dst[t[1]].x, dst[t[1]].y, 1., dst[t[2]].x, dst[t[2]].y, 1.);

                negative += cv::determinant(A)*cv::determinant(B) < 0;
            }
            if (negative != 0 && negative != 4)
                return false;
        }
        return true;
    }

    /**
     * Normalization method:
     *  - $x$ and $y$ coordinates are normalized independently
     *  - first the coordinates are shifted so that the average coordinate is \f$(0,0)\f$
     *  - then the coordinates are scaled so that the average L1 norm is 1, i.e,
     *  the average L1 norm of the \f$x\f$ coordinates is 1 and the average
     *  L1 norm of the \f$y\f$ coordinates is also 1.
     *
     * @param _m1 source points containing (X,Y), depth is CV_32F with 1 column 2 channels or
     *            2 columns 1 channel
     * @param _m2 destination points containing (x,y), depth is CV_32F with 1 column 2 channels or
     *            2 columns 1 channel
     * @param _model, CV_64FC1, 3x3, normalized, i.e., the last element is 1
     */
    int runKernel(cv::InputArray _m1, cv::InputArray _m2, cv::OutputArray _model) const override
    {
        cv::Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        int i, count = m1.checkVector(2);
        const cv::Point2f* M = m1.ptr<cv::Point2f>();
        const cv::Point2f* m = m2.ptr<cv::Point2f>();

        double LtL[9][9], W[9][1], V[9][9];
        cv::Mat _LtL( 9, 9, CV_64F, &LtL[0][0] );
        cv::Mat matW( 9, 1, CV_64F, W );
        cv::Mat matV( 9, 9, CV_64F, V );
        cv::Mat _H0( 3, 3, CV_64F, V[8] );
        cv::Mat _Htemp( 3, 3, CV_64F, V[7] );
        cv::Point2d cM(0,0), cm(0,0), sM(0,0), sm(0,0);

        for( i = 0; i < count; i++ )
        {
            cm.x += m[i].x; cm.y += m[i].y;
            cM.x += M[i].x; cM.y += M[i].y;
        }

        cm.x /= count;
        cm.y /= count;
        cM.x /= count;
        cM.y /= count;

        for( i = 0; i < count; i++ )
        {
            sm.x += fabs(m[i].x - cm.x);
            sm.y += fabs(m[i].y - cm.y);
            sM.x += fabs(M[i].x - cM.x);
            sM.y += fabs(M[i].y - cM.y);
        }

        if( fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
            fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON )
            return 0;
        sm.x = count/sm.x; sm.y = count/sm.y;
        sM.x = count/sM.x; sM.y = count/sM.y;

        double invHnorm[9] = { 1./sm.x, 0, cm.x, 0, 1./sm.y, cm.y, 0, 0, 1 };
        double Hnorm2[9] = { sM.x, 0, -cM.x*sM.x, 0, sM.y, -cM.y*sM.y, 0, 0, 1 };
        cv::Mat _invHnorm( 3, 3, CV_64FC1, invHnorm );
        cv::Mat _Hnorm2( 3, 3, CV_64FC1, Hnorm2 );

        _LtL.setTo(cv::Scalar::all(0));
        for( i = 0; i < count; i++ )
        {
            double x = (m[i].x - cm.x)*sm.x, y = (m[i].y - cm.y)*sm.y;
            double X = (M[i].x - cM.x)*sM.x, Y = (M[i].y - cM.y)*sM.y;
            double Lx[] = { X, Y, 1, 0, 0, 0, -x*X, -x*Y, -x };
            double Ly[] = { 0, 0, 0, X, Y, 1, -y*X, -y*Y, -y };
            int j, k;
            for( j = 0; j < 9; j++ )
                for( k = j; k < 9; k++ )
                    LtL[j][k] += Lx[j]*Lx[k] + Ly[j]*Ly[k];
        }
        completeSymm( _LtL );

        eigen( _LtL, matW, matV );
        _Htemp = _invHnorm*_H0;
        _H0 = _Htemp*_Hnorm2;
        _H0.convertTo(_model, _H0.type(), 1./_H0.at<double>(2,2) );

        return 1;
    }

    /**
     * Compute the reprojection error.
     * m2 = H*m1
     * @param _m1 depth CV_32F, 1-channel with 2 columns or 2-channel with 1 column
     * @param _m2 depth CV_32F, 1-channel with 2 columns or 2-channel with 1 column
     * @param _model CV_64FC1, 3x3
     * @param _err, output, CV_32FC1, square of the L2 norm
     */
    void computeError( cv::InputArray _m1, cv::InputArray _m2, cv::InputArray _model, cv::OutputArray _err ) const CV_OVERRIDE
    {
        cv::Mat m1 = _m1.getMat(), m2 = _m2.getMat(), model = _model.getMat();
        int i, count = m1.checkVector(2);
        const cv::Point2f* M = m1.ptr<cv::Point2f>();
        const cv::Point2f* m = m2.ptr<cv::Point2f>();
        const double* H = model.ptr<double>();
        float Hf[] = { (float)H[0], (float)H[1], (float)H[2], (float)H[3], (float)H[4], (float)H[5], (float)H[6], (float)H[7] };

        _err.create(count, 1, CV_32F);
        float* err = _err.getMat().ptr<float>();

        for( i = 0; i < count; i++ )
        {
            float ww = 1.f/(Hf[6]*M[i].x + Hf[7]*M[i].y + 1.f);
            float dx = (Hf[0]*M[i].x + Hf[1]*M[i].y + Hf[2])*ww - m[i].x;
            float dy = (Hf[3]*M[i].x + Hf[4]*M[i].y + Hf[5])*ww - m[i].y;
            err[i] = dx*dx + dy*dy;
        }
    }
};

class HomographyRefineCallback : public LMSolver::Callback
{
public:
    HomographyRefineCallback(cv::InputArray _src, cv::InputArray _dst)
    {
        src = _src.getMat();
        dst = _dst.getMat();
    }

    bool compute(cv::InputArray _param, cv::OutputArray _err, cv::OutputArray _Jac) const CV_OVERRIDE
    {
        int i, count = src.checkVector(2);
        cv::Mat param = _param.getMat();
        _err.create(count*2, 1, CV_64F);
        cv::Mat err = _err.getMat(), J;
        if( _Jac.needed())
        {
            _Jac.create(count*2, param.rows, CV_64F);
            J = _Jac.getMat();
            CV_Assert( J.isContinuous() && J.cols == 8 );
        }

        const cv::Point2f* M = src.ptr<cv::Point2f>();
        const cv::Point2f* m = dst.ptr<cv::Point2f>();
        const double* h = param.ptr<double>();
        double* errptr = err.ptr<double>();
        double* Jptr = J.data ? J.ptr<double>() : 0;

        for( i = 0; i < count; i++ )
        {
            double Mx = M[i].x, My = M[i].y;
            double ww = h[6]*Mx + h[7]*My + 1.;
            ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
            double xi = (h[0]*Mx + h[1]*My + h[2])*ww;
            double yi = (h[3]*Mx + h[4]*My + h[5])*ww;
            errptr[i*2] = xi - m[i].x;
            errptr[i*2+1] = yi - m[i].y;

            if( Jptr )
            {
                Jptr[0] = Mx*ww; Jptr[1] = My*ww; Jptr[2] = ww;
                Jptr[3] = Jptr[4] = Jptr[5] = 0.;
                Jptr[6] = -Mx*ww*xi; Jptr[7] = -My*ww*xi;
                Jptr[8] = Jptr[9] = Jptr[10] = 0.;
                Jptr[11] = Mx*ww; Jptr[12] = My*ww; Jptr[13] = ww;
                Jptr[14] = -Mx*ww*yi; Jptr[15] = -My*ww*yi;

                Jptr += 16;
            }
        }

        return true;
    }

    cv::Mat src, dst;
};

// Implementation based on OpenCV.
cv::Mat find_homography_extended_ransac(cv::InputArray _points1, cv::InputArray _points2,
    double ransacReprojThreshold, cv::OutputArray _mask,
    const int maxIters, const double confidence) {
  const double defaultRANSACReprojThreshold = 3;
  bool result = false;


  cv::Mat points1 = _points1.getMat(), points2 = _points2.getMat();
  cv::Mat src, dst, H, tempMask;
  int npoints = -1;

  for (int i = 1; i <= 2; i++)
  {
    cv::Mat& p = i == 1 ? points1 : points2;
    cv::Mat& m = i == 1 ? src : dst;
    npoints = p.checkVector(2, -1, false);
    if (npoints < 0 )
    {
      npoints = p.checkVector(3, -1, false);
      if (npoints < 0)
        CV_Error(cv::Error::StsBadArg, "The input arrays should be 2D or 3D point sets");
      if (npoints == 0)
        return cv::Mat();
      cv::convertPointsFromHomogeneous(p, p);
    }
    p.reshape(2, npoints).convertTo(m, CV_32F);
  }

  CV_Assert( src.checkVector(2) == dst.checkVector(2) );

  if (ransacReprojThreshold <= 0)
    ransacReprojThreshold = defaultRANSACReprojThreshold;

  cv::Ptr<BucketRANSACPointSetRegistrator::Callback> cb = cv::makePtr<HomographyEstimatorCallback>();

  if (npoints == 4) {
    tempMask = cv::Mat::ones(npoints, 1, CV_8U);
    result = cb->runKernel(src, dst, H) > 0;
  }
  else {
    result = createBucketRANSACPointSetRegistrator(cb, 4, ransacReprojThreshold, confidence, maxIters)->run(src, dst, H, tempMask);
  }

  if (result && npoints > 4) {
    compressElems(src.ptr<cv::Point2f>(), tempMask.ptr<uchar>(), 1, npoints);
    npoints = compressElems(dst.ptr<cv::Point2f>(), tempMask.ptr<uchar>(), 1, npoints);
    if (npoints > 0) {
      cv::Mat src1 = src.rowRange(0, npoints);
      cv::Mat dst1 = dst.rowRange(0, npoints);
      src = src1;
      dst = dst1;
      cb->runKernel( src, dst, H );
      cv::Mat H8(8, 1, CV_64F, H.ptr<double>());
      createLMSolver(cv::makePtr<HomographyRefineCallback>(src, dst), 10)->run(H8);
    }
  }

  if (result)
  {
    if (_mask.needed())
      tempMask.copyTo(_mask);
  }
  else
  {
    H.release();
    if (_mask.needed() ) {
      tempMask = cv::Mat::zeros(npoints >= 0 ? npoints : 0, 1, CV_8U);
      tempMask.copyTo(_mask);
    }
  }
  return H;
}
