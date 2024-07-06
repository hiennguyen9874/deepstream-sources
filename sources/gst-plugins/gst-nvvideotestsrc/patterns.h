#ifndef _PATTERNS_H_
#define _PATTERNS_H_

#ifdef __cplusplus
extern "C" {
#endif

void gst_nv_video_test_src_cuda_init(GstNvVideoTestSrc *src);
void gst_nv_video_test_src_cuda_free(GstNvVideoTestSrc *src);
void gst_nv_video_test_src_cuda_prepare(GstNvVideoTestSrc *src, NvBufSurfaceParams *surf);

void gst_nv_video_test_src_smpte(GstNvVideoTestSrc *src);
void gst_nv_video_test_src_mandelbrot(GstNvVideoTestSrc *src);
void gst_nv_video_test_src_gradient(GstNvVideoTestSrc *src);

#ifdef __cplusplus
}
#endif

#endif
