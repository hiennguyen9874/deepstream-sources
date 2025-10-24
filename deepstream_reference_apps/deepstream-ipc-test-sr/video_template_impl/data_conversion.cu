#include <cuda.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 32
#define THREADS_PER_BLOCK_1 (THREADS_PER_BLOCK - 1)

__global__ void Convert_FtFTensorKernel(float *inBuffer,
                                        unsigned char *outBuffer,
                                        unsigned int width,
                                        unsigned int height)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        int v = 255 * inBuffer[row * width + col];
        if (v < 0) {
            v = 0;
        } else if (v > 255) {
            v = 255;
        }
        outBuffer[row * width + col] = v;
    }
}

void Convert_FtFTensor(float *inBuffer,
                       unsigned char *outBuffer,
                       unsigned int width,
                       unsigned int height)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width + THREADS_PER_BLOCK_1) / threadsPerBlock.x,
                (height + THREADS_PER_BLOCK_1) / threadsPerBlock.y);

    Convert_FtFTensorKernel<<<blocks, threadsPerBlock, 0>>>(inBuffer, outBuffer, width, height);
}
