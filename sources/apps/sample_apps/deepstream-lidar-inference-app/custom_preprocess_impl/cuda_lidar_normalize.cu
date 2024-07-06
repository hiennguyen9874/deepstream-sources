#include <assert.h>
#include <cuda_runtime.h>

extern "C" void ds3dCustomCudaLidarNormalize(float *in,
                                             float *out,
                                             int points,
                                             float offset,
                                             float scale,
                                             cudaStream_t stream);

static __global__ void cuda_lidar_normalize_c4(float4 *in,
                                               float4 *out,
                                               int points,
                                               float offset,
                                               float scale)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= points)
        return;
    float4 p = in[idx];
    p.w = (p.w - offset) * scale;
    out[idx] = p;
}

extern "C" void ds3dCustomCudaLidarNormalize(float *in,
                                             float *out,
                                             int points,
                                             float offset,
                                             float scale,
                                             cudaStream_t stream)
{
    size_t threads = 64;
    size_t blocks = (points + threads - 1) / threads;
    cuda_lidar_normalize_c4<<<blocks, threads, 0, stream>>>((float4 *)in, (float4 *)out, points,
                                                            offset, scale);
}