int syncbn_sum_sqsum_cuda(
    const THCudaTensor *x, THCudaTensor *sum, THCudaTensor *sqsum);
int syncbn_forward_cuda(
    THCudaTensor *z, const THCudaTensor *x,
    const THCudaTensor *gamma, const THCudaTensor *beta,
    const THCudaTensor *mean, const THCudaTensor *var, float eps);
int syncbn_backward_xhat_cuda(
    const THCudaTensor *dz, const THCudaTensor *x,
    const THCudaTensor *mean, const THCudaTensor *var,
    THCudaTensor *sum_dz, THCudaTensor *sum_dz_xhat,
    float eps);
int syncbn_backard_cuda(
    const THCudaTensor *dz, const THCudaTensor *x,
    const THCudaTensor *gamma, const THCudaTensor *beta,
    const THCudaTensor *mean, const THCudaTensor *var,
    const THCudaTensor *sum_dz, const THCudaTensor *sum_dz_xhat,
    THCudaTensor *dx, THCudaTensor *dgamma, THCudaTensor *dbeta, float eps);
