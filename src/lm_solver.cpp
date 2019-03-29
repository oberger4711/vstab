#include "lm_solver.h"

using namespace cv;

class LMSolverImpl CV_FINAL : public LMSolver
{
public:
    LMSolverImpl() : maxIters(100) { init(); }
    LMSolverImpl(const Ptr<LMSolver::Callback>& _cb, int _maxIters) : cb(_cb), maxIters(_maxIters) { init(); }

    void init()
    {
        epsx = epsf = FLT_EPSILON;
        printInterval = 0;
    }

    int run(InputOutputArray _param0) const override
    {
        Mat param0 = _param0.getMat(), x, xd, r, rd, J, A, Ap, v, temp_d, d;
        int ptype = param0.type();

        CV_Assert( (param0.cols == 1 || param0.rows == 1) && (ptype == CV_32F || ptype == CV_64F));
        CV_Assert( cb );

        int lx = param0.rows + param0.cols - 1;
        param0.convertTo(x, CV_64F);

        if( x.cols != 1 )
            transpose(x, x);

        if( !cb->compute(x, r, J) )
            return -1;
        double S = norm(r, NORM_L2SQR);
        int nfJ = 2;

        mulTransposed(J, A, true);
        gemm(J, r, 1, noArray(), 0, v, GEMM_1_T);

        Mat D = A.diag().clone();

        const double Rlo = 0.25, Rhi = 0.75;
        double lambda = 1, lc = 0.75;
        int i, iter = 0;

        if( printInterval != 0 )
        {
            printf("************************************************************************************\n");
            printf("\titr\tnfJ\t\tSUM(r^2)\t\tx\t\tdx\t\tl\t\tlc\n");
            printf("************************************************************************************\n");
        }

        for( ;; )
        {
            CV_Assert( A.type() == CV_64F && A.rows == lx );
            A.copyTo(Ap);
            for( i = 0; i < lx; i++ )
                Ap.at<double>(i, i) += lambda*D.at<double>(i);
            solve(Ap, v, d, DECOMP_EIG);
            subtract(x, d, xd);
            if( !cb->compute(xd, rd, noArray()) )
                return -1;
            nfJ++;
            double Sd = norm(rd, NORM_L2SQR);
            gemm(A, d, -1, v, 2, temp_d);
            double dS = d.dot(temp_d);
            double R = (S - Sd)/(fabs(dS) > DBL_EPSILON ? dS : 1);

            if( R > Rhi )
            {
                lambda *= 0.5;
                if( lambda < lc )
                    lambda = 0;
            }
            else if( R < Rlo )
            {
                // find new nu if R too low
                double t = d.dot(v);
                double nu = (Sd - S)/(fabs(t) > DBL_EPSILON ? t : 1) + 2;
                nu = std::min(std::max(nu, 2.), 10.);
                if( lambda == 0 )
                {
                    invert(A, Ap, DECOMP_EIG);
                    double maxval = DBL_EPSILON;
                    for( i = 0; i < lx; i++ )
                        maxval = std::max(maxval, std::abs(Ap.at<double>(i,i)));
                    lambda = lc = 1./maxval;
                    nu *= 0.5;
                }
                lambda *= nu;
            }

            if( Sd < S )
            {
                nfJ++;
                S = Sd;
                std::swap(x, xd);
                if( !cb->compute(x, r, J) )
                    return -1;
                mulTransposed(J, A, true);
                gemm(J, r, 1, noArray(), 0, v, GEMM_1_T);
            }

            iter++;
            bool proceed = iter < maxIters && norm(d, NORM_INF) >= epsx && norm(r, NORM_INF) >= epsf;

            if( printInterval != 0 && (iter % printInterval == 0 || iter == 1 || !proceed) )
            {
                printf("%c%10d %10d %15.4e %16.4e %17.4e %16.4e %17.4e\n",
                       (proceed ? ' ' : '*'), iter, nfJ, S, x.at<double>(0), d.at<double>(0), lambda, lc);
            }

            if(!proceed)
                break;
        }

        if( param0.size != x.size )
            transpose(x, x);

        x.convertTo(param0, ptype);
        if( iter == maxIters )
            iter = -iter;

        return iter;
    }

    void setCallback(const Ptr<LMSolver::Callback>& _cb) CV_OVERRIDE { cb = _cb; }

    Ptr<LMSolver::Callback> cb;

    double epsx;
    double epsf;
    int maxIters;
    int printInterval;
};


Ptr<LMSolver> createLMSolver(const Ptr<LMSolver::Callback>& cb, int maxIters)
{
    return makePtr<LMSolverImpl>(cb, maxIters);
}
