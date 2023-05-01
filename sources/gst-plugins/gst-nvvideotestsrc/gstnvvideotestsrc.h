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

#ifndef _GST_NV_VIDEO_TEST_SRC_H_
#define _GST_NV_VIDEO_TEST_SRC_H_

#include <gst/base/base.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <nvbufsurface.h>

G_BEGIN_DECLS

#define GST_TYPE_NV_VIDEO_TEST_SRC (gst_nv_video_test_src_get_type())
#define GST_NV_VIDEO_TEST_SRC(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NV_VIDEO_TEST_SRC, GstNvVideoTestSrc))
#define GST_NV_VIDEO_TEST_SRC_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NV_VIDEO_TEST_SRC, GstNvVideoTestSrcClass))
#define GST_IS_NV_VIDEO_TEST_SRC(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NV_VIDEO_TEST_SRC))
#define GST_IS_NV_VIDEO_TEST_SRC_CLASS(obj) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NV_VIDEO_TEST_SRC))

typedef enum {
    GST_NV_VIDEO_TEST_SRC_SMPTE,
    GST_NV_VIDEO_TEST_SRC_MANDELBROT,
    GST_NV_VIDEO_TEST_SRC_GRADIENT
} GstNvVideoTestSrcPattern;

typedef enum {
    GST_NV_VIDEO_TEST_SRC_FRAMES,
    GST_NV_VIDEO_TEST_SRC_WALL_TIME,
    GST_NV_VIDEO_TEST_SRC_RUNNING_TIME
} GstNvVideoTestSrcAnimationMode;

typedef struct _GstNvVideoTestSrc GstNvVideoTestSrc;
typedef struct _GstNvVideoTestSrcClass GstNvVideoTestSrcClass;

struct _GstNvVideoTestSrc {
    GstPushSrc parent;

    // Plugin parameters.
    GstNvVideoTestSrcPattern pattern;
    GstNvVideoTestSrcAnimationMode animation_mode;
    guint gpu_id;
    NvBufSurfaceMemType memtype;
    gboolean enable_rdma;

    // Stream details set during caps negotiation.
    GstCaps *caps;
    GstVideoInfo info;

    // Runtime state.
    GstClockTime running_time;
    guint filled_frames;

    NvBufSurfaceParams *cuda_surf;
    unsigned int cuda_block_size;
    unsigned int cuda_num_blocks;
    void (*cuda_fill_image)(GstNvVideoTestSrc *src);
};

struct _GstNvVideoTestSrcClass {
    GstPushSrcClass parent_class;
};

GType gst_nv_video_test_src_get_type(void);

G_END_DECLS

#endif
