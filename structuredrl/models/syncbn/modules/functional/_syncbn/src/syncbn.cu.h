#ifndef __SYNCBN__
#define __SYNCBN__

/*
 * Exported functions
 */
extern "C" int _syncbn_sum_sqsum_cuda(int N, int C, int S, const float *x,
                                  float *sum, float *sqsum,
                                  cudaStream_t stream);
extern "C" int _syncbn_forward_cuda(
    int N, int C, int S, float *z, const float *x,
    const float *gamma, const float *beta, const float *mean, const float *var,
    float eps, cudaStream_t stream);
extern "C" int _syncbn_backward_xhat_cuda(
    int N, int C, int S, const float *dz, const float *x,
    const float *mean, const float *var, float *sum_dz, float *sum_dz_xhat,
    float eps, cudaStream_t stream);
extern "C" int _syncbn_backward_cuda(
    int N, int C, int S, const float *dz, const float *x,
    const float *gamma, const float *beta, const float *mean, const float *var,
    const float *sum_dz, const float *sum_dz_xhat,
    float *dx, float *dweight, float *dbias,
    float eps, cudaStream_t stream);


#endif
