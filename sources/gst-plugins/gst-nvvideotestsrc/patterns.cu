/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda.h>
#include <stdint.h>

#include "gstnvvideotestsrc.h"
#include "patterns.h"

#define CUDA_CALLABLE __host__ __device__

#define GREY ColorRGB(0.750, 0.750, 0.750)
#define YELLOW ColorRGB(0.750, 0.750, 0.000)
#define CYAN ColorRGB(0.000, 0.750, 0.750)
#define GREEN ColorRGB(0.000, 0.750, 0.000)
#define MAGENTA ColorRGB(0.750, 0.000, 0.750)
#define RED ColorRGB(0.750, 0.000, 0.000)
#define BLUE ColorRGB(0.000, 0.000, 0.750)
#define OXFORD ColorRGB(0.000, 0.129, 0.298)
#define WHITE ColorRGB(1.000, 1.000, 1.000)
#define VIOLET ColorRGB(0.196, 0.000, 0.416)
#define SUPERBLACK ColorRGB(0.035, 0.035, 0.035)
#define BLACK ColorRGB(0.075, 0.075, 0.075)
#define DARKGREY ColorRGB(0.114, 0.114, 0.114)

struct ColorYUV {
    CUDA_CALLABLE ColorYUV() : y(0), u(0), v(0) {}
    CUDA_CALLABLE ColorYUV(double _y, double _u, double _v) : y(_y), u(_u), v(_v) {}

    double y, u, v;
};

struct ColorRGB {
    CUDA_CALLABLE ColorRGB() : r(0), g(0), b(0) {}
    CUDA_CALLABLE ColorRGB(double _r, double _g, double _b) : r(_r), g(_g), b(_b) {}

    CUDA_CALLABLE ColorYUV toYUV() const
    {
        double y = ((0.21260 * r) + (0.71520 * g) + (0.07220 * b));
        double u = ((-0.114572 * r) + (-0.385428 * g) + (0.5 * b)) * 1.02283 + 128;
        double v = ((0.5 * r) + (-0.454153 * g) + (-0.045847 * b)) * 1.02283 + 128;
        return ColorYUV(y, u, v);
    }

    double r, g, b;
};

__device__ static void write_color(NvBufSurfaceParams *surf, int x, int y, const ColorRGB &rgb)
{
    switch (surf->colorFormat) {
    case NVBUF_COLOR_FORMAT_RGBA: {
        uint8_t *p = (uint8_t *)surf->dataPtr + (y * surf->pitch) + (x * 4);
        p[0] = (int)(rgb.r * 255);
        p[1] = (int)(rgb.g * 255);
        p[2] = (int)(rgb.b * 255);
        p[3] = 255;
        break;
    }
    case NVBUF_COLOR_FORMAT_NV12_709: {
        ColorYUV yuv = rgb.toYUV();
        uint8_t *p = (uint8_t *)surf->dataPtr + (y * surf->pitch) + x;
        *p = (int)(yuv.y * 255);

        if (x % 2 == 0 && y % 2 == 0) {
            uint8_t *uv = (uint8_t *)surf->dataPtr + surf->planeParams.offset[1] +
                          ((y / 2) * surf->planeParams.pitch[1]) + x;
            uv[0] = (int)(yuv.u * 255);
            uv[1] = (int)(yuv.v * 255);
        }
        break;
    }
    case NVBUF_COLOR_FORMAT_YUV420_709: {
        ColorYUV yuv = rgb.toYUV();
        uint8_t *p = (uint8_t *)surf->dataPtr + (y * surf->pitch) + x;
        *p = (int)(yuv.y * 255);

        if (x % 2 == 0 && y % 2 == 0) {
            uint8_t *u = (uint8_t *)surf->dataPtr + surf->planeParams.offset[1] +
                         ((y / 2) * surf->planeParams.pitch[1]) + (x / 2);
            *u = (int)(yuv.u * 255);
            uint8_t *v = (uint8_t *)surf->dataPtr + surf->planeParams.offset[2] +
                         ((y / 2) * surf->planeParams.pitch[2]) + (x / 2);
            *v = (int)(yuv.v * 255);
        }
        break;
    }
    }
}

__global__ void smpte_kernel(NvBufSurfaceParams *surf)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < surf->width * surf->height; i += stride) {
        int p_y = i / surf->width;
        int p_x = i - (p_y * surf->width);
        ColorRGB c;
        if (p_y < 0.67 * surf->height) {
            int bar_width = surf->width / 7.0;
            if (p_x < bar_width)
                c = GREY;
            else if (p_x < bar_width * 2)
                c = YELLOW;
            else if (p_x < bar_width * 3)
                c = CYAN;
            else if (p_x < bar_width * 4)
                c = GREEN;
            else if (p_x < bar_width * 5)
                c = MAGENTA;
            else if (p_x < bar_width * 6)
                c = RED;
            else
                c = BLUE;
        } else if (p_y < 0.75 * surf->height) {
            int bar_width = surf->width / 7.0;
            if (p_x < bar_width)
                c = BLUE;
            else if (p_x < bar_width * 2)
                c = BLACK;
            else if (p_x < bar_width * 3)
                c = MAGENTA;
            else if (p_x < bar_width * 4)
                c = BLACK;
            else if (p_x < bar_width * 5)
                c = CYAN;
            else if (p_x < bar_width * 6)
                c = BLACK;
            else
                c = GREY;
        } else {
            int bar_width = (surf->width / 7.0 * 5.0) / 4.0;
            if (p_x < bar_width)
                c = OXFORD;
            else if (p_x < bar_width * 2)
                c = WHITE;
            else if (p_x < bar_width * 3)
                c = VIOLET;
            else if (p_x < bar_width * 4)
                c = BLACK;
            else if (p_x < (surf->width / 21.0) * 16)
                c = SUPERBLACK;
            else if (p_x < (surf->width / 21.0) * 17)
                c = BLACK;
            else if (p_x < (int)(surf->width / 7.0) * 6)
                c = DARKGREY;
            else
                c = BLACK;
        }
        write_color(surf, p_x, p_y, c);
    }
}

extern "C" void gst_nv_video_test_src_smpte(GstNvVideoTestSrc *src)
{
    smpte_kernel<<<src->cuda_num_blocks, src->cuda_block_size>>>(src->cuda_surf);
}

__device__ static int mandelbrot(double x, double y, int max_iter)
{
    double a = 0, b = 0, asq = 0, bsq = 0;

    int i = 0;
    while (i++ < max_iter) {
        b = (a * b) * 2 + y;
        a = asq - bsq + x;
        asq = a * a;
        bsq = b * b;

        if (asq + bsq > 4.0)
            break;
    }

    return i;
}

__global__ void mandelbrot_kernel(NvBufSurfaceParams *surf,
                                  ColorRGB *colors,
                                  int num_colors,
                                  double x_off,
                                  double y_off,
                                  double scale)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < surf->width * surf->height; i += stride) {
        int p_y = i / surf->width;
        int p_x = i - (p_y * surf->width);
        double x = (((double)p_x / surf->width) * 2 - 1) * scale + x_off;
        double y = (((double)p_y / surf->height) * 2 - 1) * scale + y_off;
        ColorRGB &c = colors[mandelbrot(x, y, num_colors)];
        write_color(surf, p_x, p_y, c);
    }
}

extern "C" void gst_nv_video_test_src_mandelbrot(GstNvVideoTestSrc *src)
{
    // Initialize the static color array on the first call.
    static const int MAX_ITER = 100;
    static ColorRGB *colors = NULL;
    if (!colors) {
        ColorRGB host_colors[MAX_ITER + 1];
        for (int i = 0; i <= MAX_ITER; i++) {
            float freq = 6.3 / MAX_ITER;
            double r = (sin(freq * i + 5) + 1.0) / 2.0;
            double g = (sin(freq * i + 4) + 1.0) / 2.0;
            double b = (sin(freq * i + 3) + 1.0) / 2.0;
            host_colors[i] = ColorRGB(r, g, b);
        }
        host_colors[MAX_ITER] = ColorRGB(0, 0, 0);

        int size = sizeof(ColorRGB) * (MAX_ITER + 1);
        cudaMalloc((void **)&colors, size);
        cudaMemcpy(colors, host_colors, size, cudaMemcpyHostToDevice);
    }

    const double MIN_SCALE = 0.000001;
    const double MAX_SCALE = 2.0;
    const double X_OFF = -0.734072;
    const double Y_OFF = 0.248116;
    const double log_min = log(MIN_SCALE);
    const double log_max = log(MAX_SCALE);

    // Determine the zoom/scale.
    double interp;
    if (src->animation_mode == GST_NV_VIDEO_TEST_SRC_FRAMES) {
        const uint32_t ZOOM_SPEED = 500;
        interp = (src->filled_frames / ZOOM_SPEED) % 2
                     ? (src->filled_frames % ZOOM_SPEED) / (double)ZOOM_SPEED
                     : (ZOOM_SPEED - (src->filled_frames % ZOOM_SPEED)) / (double)ZOOM_SPEED;
    } else {
        const uint32_t ZOOM_SPEED = 10000000;
        guint time;
        if (src->animation_mode == GST_NV_VIDEO_TEST_SRC_WALL_TIME)
            time = g_get_real_time();
        else
            time = src->running_time / 1000;
        interp = (time / ZOOM_SPEED) % 2 ? (time % ZOOM_SPEED) / (double)ZOOM_SPEED
                                         : (ZOOM_SPEED - (time % ZOOM_SPEED)) / (double)ZOOM_SPEED;
    }
    double log_scale = log_min + (log_max - log_min) * interp;
    double scale = exp(log_scale);

    mandelbrot_kernel<<<src->cuda_num_blocks, src->cuda_block_size>>>(
        src->cuda_surf, colors, MAX_ITER, X_OFF, Y_OFF, scale);
}

__global__ void gradient_kernel(NvBufSurfaceParams *surf)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < surf->width * surf->height; i += stride) {
        int p_y = i / surf->width;
        int p_x = i - (p_y * surf->width);
        double x = (double)p_x / surf->width;
        double y = (double)p_y / surf->height;
        ColorRGB c(x, y, (x + y) / 2.0);
        write_color(surf, p_x, p_y, c);
    }
}

extern "C" void gst_nv_video_test_src_gradient(GstNvVideoTestSrc *src)
{
    gradient_kernel<<<src->cuda_num_blocks, src->cuda_block_size>>>(src->cuda_surf);
}

extern "C" void gst_nv_video_test_src_cuda_init(GstNvVideoTestSrc *src)
{
    if (src->cuda_surf)
        cudaFree(src->cuda_surf);

    // The details of the surface to be rendered into by CUDA are provided each
    // frame by the NvBufSurfaceParams. Since these details are read by the CUDA
    // kernel, we need to copy this struct into CUDA-accessible memory. We
    // allocate CUDA memory for this structure here.
    cudaMalloc(&src->cuda_surf, sizeof(NvBufSurfaceParams));
    src->cuda_block_size = 512;
    src->cuda_num_blocks =
        (src->info.width * src->info.height + src->cuda_block_size - 1) / src->cuda_block_size;
}

extern "C" void gst_nv_video_test_src_cuda_free(GstNvVideoTestSrc *src)
{
    cudaFree(src->cuda_surf);
}

extern "C" void gst_nv_video_test_src_cuda_prepare(GstNvVideoTestSrc *src, NvBufSurfaceParams *surf)
{
    // Copy the details of the surface that we're about to render into
    // to the CUDA-accessible copy.
    cudaMemcpy(src->cuda_surf, surf, sizeof(NvBufSurfaceParams), cudaMemcpyHostToDevice);
}
