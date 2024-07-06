#ifndef __NVDS_SEQUENCE_PREPROCESS_KERNEL_H__
#define __NVDS_SEQUENCE_PREPROCESS_KERNEL_H__

#include <cuda.h>
#include <cuda_runtime.h>

#define VEC4_SIZE 4
// float vector structure for multiple channels
typedef struct {
    float d[VEC4_SIZE];
} Float4Vec;

/**
 * NCDHW preprocess per ROI image
 *
 * @param outPtr output data pointer offset to current image position
 * @param inPtr input data pointer
 */
cudaError_t preprocessNCDHW(void *outPtr,
                            unsigned int outC,
                            unsigned int H,
                            unsigned int W,
                            unsigned int S,
                            const void *inPtr,
                            unsigned int inC,
                            unsigned int inRowPitch,
                            Float4Vec scales,
                            Float4Vec means,
                            bool swapRB,
                            cudaStream_t stream);

#endif // __NVDS_SEQUENCE_PREPROCESS_KERNEL_H__