#ifndef DRAW_BBOX_H
#define DRAW_BBOX_H

#include <cstdint>
// #include <stdint.h>
#include <cuda_runtime.h>

void drawBoundingBox_cuda_unit_alpha(uint8_t *pBGRA,
                                     const int nWidth,
                                     const int nHeight,
                                     const int nBgraPitch,
                                     const int x_min,
                                     const int y_min,
                                     const int x_max,
                                     const int y_max,
                                     cudaStream_t stream,
                                     unsigned int r,
                                     unsigned int g,
                                     unsigned int b,
                                     unsigned int border_width);

void drawBoundingBox_cuda(uint8_t *pBGRA,
                          const int nWidth,
                          const int nHeight,
                          const int nBgraPitch,
                          const int x_min,
                          const int y_min,
                          const int x_max,
                          const int y_max,
                          cudaStream_t stream,
                          unsigned int r,
                          unsigned int g,
                          unsigned int b,
                          float a,
                          unsigned int border_width);

void bboxAlphaFill_cuda(uint8_t *pBGRA,
                        const int nWidth,
                        const int nHeight,
                        const int nBgraPitch,
                        const int x_min,
                        const int y_min,
                        const int x_max,
                        const int y_max,
                        unsigned int r,
                        unsigned int g,
                        unsigned int b,
                        float a,
                        cudaStream_t stream);

void bgra_to_gray(uint8_t *dpBgra,
                  const int nBgraPitch,
                  uint8_t *dpGray,
                  const int nGrayPitch,
                  const int nWidth,
                  const int nHeight,
                  cudaStream_t stream);

void nv12_to_gray_batch(const uint8_t *pNv12,
                        int nNv12Pitch,
                        uint8_t *pGray,
                        int nGrayPitch,
                        int nWidth,
                        int nHeight,
                        int nBatchSize,
                        cudaStream_t stream);

#endif // DRAW_BBOX_H
