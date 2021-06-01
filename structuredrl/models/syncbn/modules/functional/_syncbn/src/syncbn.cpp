// All functions assume that input and output tensors are already initialized
// and have the correct dimensions
#include <THC/THC.h>

extern THCState *state;

void get_sizes(const THCudaTensor *t, int *N, int *C, int *S) {
    // Get sizes
    *S = 1;
    *N = THCudaTensor_size(state, t, 0);
    *C = THCudaTensor_size(state, t, 1);
    if (THCudaTensor_nDimension(state, t) > 2) {
        for (int i = 2; i < THCudaTensor_nDimension(state, t); ++i) {
            *S *= THCudaTensor_size(state, t, i);
        }
    }
}

// Forward definition of implementation functions
extern "C" {
    int _syncbn_sum_sqsum_cuda(int N, int C, int S,
                           const float *x, float *sum, float *sqsum,
                           cudaStream_t stream);
    int _syncbn_forward_cuda(
        int N, int C, int S, float *z, const float *x,
        const float *gamma, const float *beta,
        const float *mean, const float *var, float eps, cudaStream_t stream);
    int _syncbn_backward_xhat_cuda(
        int N, int C, int S, const float *dz, const float *x,
        const float *mean, const float *var, float *sum_dz, float *sum_dz_xhat,
        float eps, cudaStream_t stream);
    int _syncbn_backward_cuda(
        int N, int C, int S, const float *dz, const float *x,
        const float *gamma, const float *beta,
        const float *mean, const float *var,
        const float *sum_dz, const float *sum_dz_xhat,
        float *dx, float *dgamma, float *dbeta,
        float eps, cudaStream_t stream);
}

extern "C" int syncbn_sum_sqsum_cuda(
    const THCudaTensor *x, THCudaTensor *sum, THCudaTensor *sqsum) {
    cudaStream_t stream = THCState_getCurrentStream(state);

    int S, N, C;
    get_sizes(x, &N, &C, &S);

    // Get pointers
    const float *x_data = THCudaTensor_data(state, x);
    float *sum_data = THCudaTensor_data(state, sum);
    float *sqsum_data = THCudaTensor_data(state, sqsum);

    return _syncbn_sum_sqsum_cuda(N, C, S, x_data, sum_data, sqsum_data, stream);
}

extern "C" int syncbn_forward_cuda(
    THCudaTensor *z, const THCudaTensor *x,
    const THCudaTensor *gamma, const THCudaTensor *beta,
    const THCudaTensor *mean, const THCudaTensor *var, float eps){
    cudaStream_t stream = THCState_getCurrentStream(state);

    int S, N, C;
    get_sizes(x, &N, &C, &S);

    // Get pointers
    float *z_data = THCudaTensor_data(state, z);
    const float *x_data = THCudaTensor_data(state, x);
    const float *gamma_data = THCudaTensor_nDimension(state, gamma) != 0 ?
                               THCudaTensor_data(state, gamma) : 0;
    const float *beta_data = THCudaTensor_nDimension(state, beta) != 0 ?
                             THCudaTensor_data(state, beta) : 0;
    const float *mean_data = THCudaTensor_data(state, mean);
    const float *var_data = THCudaTensor_data(state, var);

    return _syncbn_forward_cuda(
        N, C, S, z_data, x_data, gamma_data, beta_data,
        mean_data, var_data, eps, stream);

}

extern "C" int syncbn_backward_xhat_cuda(
    const THCudaTensor *dz, const THCudaTensor *x,
    const THCudaTensor *mean, const THCudaTensor *var,
    THCudaTensor *sum_dz, THCudaTensor *sum_dz_xhat, float eps) {
    cudaStream_t stream = THCState_getCurrentStream(state);

    int S, N, C;
    get_sizes(dz, &N, &C, &S);

    // Get pointers
    const float *dz_data = THCudaTensor_data(state, dz);
    const float *x_data = THCudaTensor_data(state, x);
    const float *mean_data = THCudaTensor_data(state, mean);
    const float *var_data = THCudaTensor_data(state, var);
    float *sum_dz_data = THCudaTensor_data(state, sum_dz);
    float *sum_dz_xhat_data = THCudaTensor_data(state, sum_dz_xhat);

    return _syncbn_backward_xhat_cuda(
        N, C, S, dz_data, x_data, mean_data, var_data,
        sum_dz_data, sum_dz_xhat_data, eps, stream);

}
extern "C" int syncbn_backard_cuda(
    const THCudaTensor *dz, const THCudaTensor *x,
    const THCudaTensor *gamma, const THCudaTensor *beta,
    const THCudaTensor *mean, const THCudaTensor *var,
    const THCudaTensor *sum_dz, const THCudaTensor *sum_dz_xhat,
    THCudaTensor *dx, THCudaTensor *dgamma, THCudaTensor *dbeta, float eps) {
    cudaStream_t stream = THCState_getCurrentStream(state);

    int S, N, C;
    get_sizes(dz, &N, &C, &S);

    // Get pointers
    const float *dz_data = THCudaTensor_data(state, dz);
    const float *x_data = THCudaTensor_data(state, x);
    const float *gamma_data = THCudaTensor_nDimension(state, gamma) != 0 ?
                               THCudaTensor_data(state, gamma) : 0;
    const float *beta_data = THCudaTensor_nDimension(state, beta) != 0 ?
                             THCudaTensor_data(state, beta) : 0;
    const float *mean_data = THCudaTensor_data(state, mean);
    const float *var_data = THCudaTensor_data(state, var);
    const float *sum_dz_data = THCudaTensor_data(state, sum_dz);
    const float *sum_dz_xhat_data = THCudaTensor_data(state, sum_dz_xhat);
    float *dx_data = THCudaTensor_nDimension(state, dx) != 0 ?
                     THCudaTensor_data(state, dx) : 0;
    float *dgamma_data = THCudaTensor_nDimension(state, dgamma) != 0 ?
                          THCudaTensor_data(state, dgamma) : 0;
    float *dbeta_data = THCudaTensor_nDimension(state, dbeta) != 0 ?
                        THCudaTensor_data(state, dbeta) : 0;

    return _syncbn_backward_cuda(
        N, C, S, dz_data, x_data, gamma_data, beta_data,
        mean_data, var_data, sum_dz_data, sum_dz_xhat_data,
        dx_data, dgamma_data, dbeta_data, eps, stream);
}