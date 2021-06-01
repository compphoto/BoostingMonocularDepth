#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "common.h"
#include "syncbn.cu.h"

/*
 * Device functions and data structures
 */
struct Float2 {
  float v1, v2;
  __device__ Float2() {}
  __device__ Float2(float _v1, float _v2) : v1(_v1), v2(_v2) {}
  __device__ Float2(float v) : v1(v), v2(v) {}
  __device__ Float2(int v) : v1(v), v2(v) {}
  __device__ Float2 &operator+=(const Float2 &a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

struct GradOp {
  __device__ GradOp(float _gamma, float _beta, const float *_z,
   const float *_dz, int c, int s)
      : gamma(_gamma), beta(_beta), z(_z), dz(_dz), C(c), S(s) {}
  __device__ __forceinline__ Float2 operator()(int batch, int plane, int n) {
    float _y = (z[(batch * C + plane) * S + n] - beta) / gamma;
    float _dz = dz[(batch * C + plane) * S + n];
    return Float2(_dz, _y * _dz);
  }
  const float gamma;
  const float beta;
  const float *z;
  const float *dz;
  const int C;
  const int S;
};

static __device__ __forceinline__ float warpSum(float val) {
#if __CUDA_ARCH__ >= 300
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE);
  }
#else
  __shared__ float values[MAX_BLOCK_SIZE];
  values[threadIdx.x] = val;
  __threadfence_block();
  const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  for (int i = 1; i < WARP_SIZE; i++) {
    val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
  }
#endif
  return val;
}

static __device__ __forceinline__ Float2 warpSum(Float2 value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

template <typename T, typename Op>
__device__ T reduce(Op op, int plane, int N, int C, int S) {
  T sum = (T)0;
  for (int batch = 0; batch < N; ++batch) {
    for (int x = threadIdx.x; x < S; x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];
  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    shared[threadIdx.x / WARP_SIZE] = sum;
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T)0;
  }
  __syncthreads();
  if (threadIdx.x / WARP_SIZE == 0) {
    sum = warpSum(shared[threadIdx.x]);
    if (threadIdx.x == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

/*----------------------------------------------------------------------------
 *
 * BatchNorm2dSyncFunc Kernel implementations
 *
 *---------------------------------------------------------------------------*/

struct SqSumOp {
  __device__ SqSumOp(const float *t, int c, int s)
      : tensor(t), C(c), S(s) {}
  __device__ __forceinline__ Float2 operator()(int batch, int plane, int n) {
    float t = tensor[(batch * C + plane) * S + n];
    return Float2(t, t * t);
  }
  const float *tensor;
  const int C;
  const int S;
};

struct XHatOp {
  __device__ XHatOp(float _gamma, float _beta, const float *_z,
   const float *_dz, int c, int s)
      : gamma(_gamma), beta(_beta), z(_z), dz(_dz), C(c), S(s) {}
  __device__ __forceinline__ Float2 operator()(int batch, int plane, int n) {
    // xhat = (x-beta)*gamma
    float _xhat = (z[(batch * C + plane) * S + n] - beta) * gamma;
    // for dxhat*x_hat
    float _dz = dz[(batch * C + plane) * S + n];
    return Float2(_dz, _dz * _xhat);
  }
  const float gamma;
  const float beta;
  const float *z;
  const float *dz;
  const int C;
  const int S;
};

__global__ void syncbn_sum_sqsum_kernel(const float *x, float *sum, float *sqsum,
                                int N, int C, int S) {
  int plane = blockIdx.x;
  Float2 res = reduce<Float2, SqSumOp>(SqSumOp(x, C, S), plane, N, C, S);
  float _sum = res.v1;
  float _sqsum = res.v2;
  __syncthreads();
  if (threadIdx.x == 0) {
    sum[plane] = _sum;
    sqsum[plane] = _sqsum;
  }
}

__global__ void syncbn_forward_kernel(
        float *z, const float *x, const float *gamma, const float *beta,
        const float *mean, const float *var, float eps, int N, int C, int S) {

    int c = blockIdx.x;
    float _mean = mean[c];
    float _var = var[c];
    float invtsd = 0;
    if (_var != 0.f || eps != 0.f) {
      invtsd = 1 / sqrt(_var + eps);
    }
    float _gamma = gamma != 0 ? gamma[c] : 1.f;
    float _beta = beta != 0 ? beta[c] : 0.f;
    for (int batch = 0; batch < N; ++batch) {
      for (int n = threadIdx.x; n < S; n += blockDim.x) {
        float _x = x[(batch * C + c) * S + n];
        float _xhat = (_x - _mean) * invtsd;
        float _z = _xhat * _gamma + _beta;
        z[(batch * C + c) * S + n] = _z;
      }
    }
}

__global__ void syncbn_backward_xhat_kernel(
        const float *dz, const float *x, const float *mean, const float *var,
        float *sum_dz, float *sum_dz_xhat, float eps, int N, int C, int S) {

    int c = blockIdx.x;
    float _mean = mean[c];
    float _var = var[c];
    float _invstd = 0;
    if (_var != 0.f || eps != 0.f) {
          _invstd = 1 / sqrt(_var + eps);
    }
    Float2 res = reduce<Float2, XHatOp>(
        XHatOp(_invstd, _mean, x, dz, C, S), c, N, C, S);
    // \sum(\frac{dJ}{dy_i})
    float _sum_dz = res.v1;
    // \sum(\frac{dJ}{dy_i}*\hat{x_i})
    float _sum_dz_xhat = res.v2;
    __syncthreads();
    if (threadIdx.x == 0) {
        // \sum(\frac{dJ}{dy_i})
        sum_dz[c] = _sum_dz;
        // \sum(\frac{dJ}{dy_i}*\hat{x_i})
        sum_dz_xhat[c] = _sum_dz_xhat;
    }
}


__global__ void syncbn_backward_kernel(
        const float *dz, const float *x, const float *gamma, const float *beta,
        const float *mean, const float *var,
        const float *sum_dz, const float *sum_dz_xhat,
        float *dx, float *dgamma, float *dbeta,
        float eps, int N, int C, int S) {

    int c = blockIdx.x;
    float _mean = mean[c];
    float _var = var[c];
    float _gamma = gamma != 0 ? gamma[c] : 1.f;
    float _sum_dz = sum_dz[c];
    float _sum_dz_xhat = sum_dz_xhat[c];
    float _invstd = 0;
    if (_var != 0.f || eps != 0.f) {
          _invstd = 1 / sqrt(_var + eps);
    }
    /*
      \frac{dJ}{dx_i} = \frac{1}{N\sqrt{(\sigma^2+\epsilon)}} (
        N\frac{dJ}{d\hat{x_i}} -
        \sum_{j=1}^{N}(\frac{dJ}{d\hat{x_j}}) -
        \hat{x_i}\sum_{j=1}^{N}(\frac{dJ}{d\hat{x_j}}\hat{x_j})
      )
      Note : N is omitted here since it will be accumulated and
      _sum_dz and _sum_dz_xhat expected to be already normalized
      before the call.
    */
    if (dx != 0) {
        float _mul = _gamma * _invstd;
        for (int batch = 0; batch < N; ++batch) {
            for (int n = threadIdx.x; n < S; n += blockDim.x) {
                float _dz = dz[(batch * C + c) * S + n];
                float _xhat = (x[(batch * C + c) * S + n] - _mean) * _invstd;
                float _dx = (_dz - _sum_dz - _xhat * _sum_dz_xhat) * _mul;
                dx[(batch * C + c) * S + n] = _dx;
            }
        }
    }
    float _norm = N * S;
    if (dgamma != 0) {
        if (threadIdx.x == 0) {
            // \frac{dJ}{d\gamma} = \sum(\frac{dJ}{dy_i}*\hat{x_i})
            dgamma[c] += _sum_dz_xhat * _norm;
        }
    }
    if (dbeta != 0) {
        if (threadIdx.x == 0) {
            // \frac{dJ}{d\beta} = \sum(\frac{dJ}{dy_i})
            dbeta[c] += _sum_dz * _norm;
        }
    }
}

extern "C" int _syncbn_sum_sqsum_cuda(int N, int C, int S,
                                  const float *x, float *sum, float *sqsum,
                                  cudaStream_t stream) {
  // Run kernel
  dim3 blocks(C);
  dim3 threads(getNumThreads(S));
  syncbn_sum_sqsum_kernel<<<blocks, threads, 0, stream>>>(x, sum, sqsum, N, C, S);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

extern "C" int _syncbn_forward_cuda(
    int N, int C, int S, float *z, const float *x,
    const float *gamma, const float *beta, const float *mean, const float *var,
    float eps, cudaStream_t stream) {

    // Run kernel
    dim3 blocks(C);
    dim3 threads(getNumThreads(S));
    syncbn_forward_kernel<<<blocks, threads, 0, stream>>>(
        z, x, gamma, beta, mean, var, eps, N, C, S);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return 0;
    else
        return 1;
}


extern "C" int _syncbn_backward_xhat_cuda(
    int N, int C, int S, const float *dz, const float *x,
    const float *mean, const float *var, float *sum_dz, float *sum_dz_xhat,
    float eps, cudaStream_t stream) {

    // Run kernel
    dim3 blocks(C);
    dim3 threads(getNumThreads(S));
    syncbn_backward_xhat_kernel<<<blocks, threads, 0, stream>>>(
        dz, x,mean, var, sum_dz, sum_dz_xhat, eps, N, C, S);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return 0;
    else
        return 1;
}


extern "C" int _syncbn_backward_cuda(
    int N, int C, int S, const float *dz, const float *x,
    const float *gamma, const float *beta, const float *mean, const float *var,
    const float *sum_dz, const float *sum_dz_xhat,
    float *dx, float *dgamma, float *dbeta, float eps, cudaStream_t stream) {

    // Run kernel
    dim3 blocks(C);
    dim3 threads(getNumThreads(S));
    syncbn_backward_kernel<<<blocks, threads, 0, stream>>>(
        dz, x, gamma, beta, mean, var, sum_dz, sum_dz_xhat,
        dx, dgamma, dbeta, eps, N, C, S);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return 0;
    else
        return 1;
}

