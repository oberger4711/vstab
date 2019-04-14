#include "registration.h"

#include "opencv2/opencv_modules.hpp"

// This is based on lkpyramid.cpp from OpenCV.

static void
getRTMatrix(const std::vector<cv::Point2f> a, const std::vector<cv::Point2f> b,
             int count, cv::Mat& M, bool fullAffine)
{
    CV_Assert(M.isContinuous());

    if(fullAffine)
    {
        double sa[6][6]={{0.}}, sb[6]={0.};
        cv::Mat A(6, 6, CV_64F, &sa[0][0]), B(6, 1, CV_64F, sb);
        cv::Mat MM = M.reshape(1, 6);

        for (int i = 0; i < count; i++)
        {
            sa[0][0] += a[i].x*a[i].x;
            sa[0][1] += a[i].y*a[i].x;
            sa[0][2] += a[i].x;

            sa[1][1] += a[i].y*a[i].y;
            sa[1][2] += a[i].y;

            sb[0] += a[i].x*b[i].x;
            sb[1] += a[i].y*b[i].x;
            sb[2] += b[i].x;
            sb[3] += a[i].x*b[i].y;
            sb[4] += a[i].y*b[i].y;
            sb[5] += b[i].y;
        }

        sa[3][4] = sa[4][3] = sa[1][0] = sa[0][1];
        sa[3][5] = sa[5][3] = sa[2][0] = sa[0][2];
        sa[4][5] = sa[5][4] = sa[2][1] = sa[1][2];

        sa[3][3] = sa[0][0];
        sa[4][4] = sa[1][1];
        sa[5][5] = sa[2][2] = count;

        cv::solve(A, B, MM, cv::DECOMP_EIG);
    }
    else
    {
        double sa[4][4]={{0.}}, sb[4]={0.}, m[4] = {0};
        cv::Mat A(4, 4, CV_64F, sa), B(4, 1, CV_64F, sb);
        cv::Mat MM(4, 1, CV_64F, m);

        for (int i = 0; i < count; i++)
        {
            sa[0][0] += a[i].x*a[i].x + a[i].y*a[i].y;
            sa[0][2] += a[i].x;
            sa[0][3] += a[i].y;

            sb[0] += a[i].x*b[i].x + a[i].y*b[i].y;
            sb[1] += a[i].x*b[i].y - a[i].y*b[i].x;
            sb[2] += b[i].x;
            sb[3] += b[i].y;
        }

        sa[1][1] = sa[0][0];
        sa[2][1] = sa[1][2] = -sa[0][3];
        sa[3][1] = sa[1][3] = sa[2][0] = sa[0][2];
        sa[2][2] = sa[3][3] = count;
        sa[3][0] = sa[0][3];

        cv::solve(A, B, MM, cv::DECOMP_EIG);

        double* om = M.ptr<double>();
        om[0] = om[4] = m[0];
        om[1] = -m[1];
        om[3] = m[1];
        om[2] = m[2];
        om[5] = m[3];
    }
}

cv::Mat estimateRigidTransform_extended(cv::InputArray src1, cv::InputArray src2, bool fullAffine)
{
    return estimateRigidTransform_extended(src1, src2, fullAffine, 500, 0.5, 3);
}

cv::Mat estimateRigidTransform_extended(cv::InputArray src1, cv::InputArray src2, bool fullAffine, int ransacMaxIters, double ransacGoodRatio,
                                    const int ransacSize0)
{
    cv::Mat M(2, 3, CV_64F), A = src1.getMat(), B = src2.getMat();

    const int COUNT = 15;
    const int WIDTH = 160, HEIGHT = 120;

    std::vector<cv::Point2f> pA, pB;
    std::vector<int> good_idx;
    std::vector<uchar> status;

    double scale = 1.;
    int i, j, k, k1;

    cv::RNG rng((uint64)-1);
    int good_count = 0;

    assert(ransacSize0 < 3 && "ransacSize0 should have value bigger than 2.");
    assert((ransacGoodRatio > 1 || ransacGoodRatio < 0) && "ransacGoodRatio should have value between 0 and 1");
    assert((A.size() != B.size()) && "Both input images must have the same size");
    assert((A.type() != B.type()) && "Both input images must have the same data type");

    int count = A.checkVector(2);

    assert(count > 0);
    A.reshape(2, count).convertTo(pA, CV_32F);
    B.reshape(2, count).convertTo(pB, CV_32F);

    good_idx.resize(count);

    if(count < ransacSize0)
        return cv::Mat();

    cv::Rect brect = boundingRect(pB);

    std::vector<cv::Point2f> a(ransacSize0);
    std::vector<cv::Point2f> b(ransacSize0);

    // RANSAC stuff:
    // 1. find the consensus
    for (k = 0; k < ransacMaxIters; k++)
    {
        std::vector<int> idx(ransacSize0);
        // choose random 3 non-complanar points from A & B
        for (i = 0; i < ransacSize0; i++)
        {
            for (k1 = 0; k1 < ransacMaxIters; k1++)
            {
                idx[i] = rng.uniform(0, count);

                for (j = 0; j < i; j++)
                {
                    if(idx[j] == idx[i])
                        break;
                    // check that the points are not very close one each other
                    if(fabs(pA[idx[i]].x - pA[idx[j]].x) +
                        fabs(pA[idx[i]].y - pA[idx[j]].y) < FLT_EPSILON)
                        break;
                    if(fabs(pB[idx[i]].x - pB[idx[j]].x) +
                        fabs(pB[idx[i]].y - pB[idx[j]].y) < FLT_EPSILON)
                        break;
                }

                if(j < i)
                    continue;

                if(i+1 == ransacSize0)
                {
                    // additional check for non-complanar vectors
                    a[0] = pA[idx[0]];
                    a[1] = pA[idx[1]];
                    a[2] = pA[idx[2]];

                    b[0] = pB[idx[0]];
                    b[1] = pB[idx[1]];
                    b[2] = pB[idx[2]];

                    double dax1 = a[1].x - a[0].x, day1 = a[1].y - a[0].y;
                    double dax2 = a[2].x - a[0].x, day2 = a[2].y - a[0].y;
                    double dbx1 = b[1].x - b[0].x, dby1 = b[1].y - b[0].y;
                    double dbx2 = b[2].x - b[0].x, dby2 = b[2].y - b[0].y;
                    const double eps = 0.01;

                    if(fabs(dax1*day2 - day1*dax2) < eps*std::sqrt(dax1*dax1+day1*day1)*std::sqrt(dax2*dax2+day2*day2) ||
                        fabs(dbx1*dby2 - dby1*dbx2) < eps*std::sqrt(dbx1*dbx1+dby1*dby1)*std::sqrt(dbx2*dbx2+dby2*dby2))
                        continue;
                }
                break;
            }

            if(k1 >= ransacMaxIters)
                break;
        }

        if(i < ransacSize0)
            continue;

        // estimate the transformation using 3 points
        getRTMatrix(a, b, 3, M, fullAffine);

        const double* m = M.ptr<double>();
        for (i = 0, good_count = 0; i < count; i++)
        {
            if(std::abs(m[0]*pA[i].x + m[1]*pA[i].y + m[2] - pB[i].x) +
                std::abs(m[3]*pA[i].x + m[4]*pA[i].y + m[5] - pB[i].y) < std::max(brect.width,brect.height)*0.05)
                good_idx[good_count++] = i;
        }

        if(good_count >= count*ransacGoodRatio)
            break;
    }

    if(k >= ransacMaxIters)
        return cv::Mat();

    if(good_count < count)
    {
        for (i = 0; i < good_count; i++)
        {
            j = good_idx[i];
            pA[i] = pA[j];
            pB[i] = pB[j];
        }
    }

    getRTMatrix(pA, pB, good_count, M, fullAffine);
    M.at<double>(0, 2) /= scale;
    M.at<double>(1, 2) /= scale;

    return M;
}
