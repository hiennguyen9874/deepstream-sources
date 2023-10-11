/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights
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

#ifndef __GST_NVSTREAMDEMUX_H__
#define __GST_NVSTREAMDEMUX_H__

#include <gst/gst.h>

G_BEGIN_DECLS
#define GST_TYPE_NVSTREAMDEMUX (gst_nvstreamdemux_2_get_type())
#define GST_NVSTREAMDEMUX(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVSTREAMDEMUX, GstNvStreamDemux))
#define GST_NVSTREAMDEMUX_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVSTREAMDEMUX, GstNvStreamDemuxClass))
#define GST_IS_NVSTREAMDEMUX(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVSTREAMDEMUX))
#define GST_IS_NVSTREAMDEMUX_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVSTREAMDEMUX))
typedef struct _GstNvStreamDemux GstNvStreamDemux;
typedef struct _GstNvStreamDemuxClass GstNvStreamDemuxClass;

struct _GstNvStreamDemux {
    GstElement element;

    GstPad *sinkpad;

    GHashTable *pad_indexes;
    GHashTable *pad_framerates;
    GHashTable *pad_caps_is_raw;
    GHashTable *pad_stream_start_sent;
    GHashTable *eos_flag;

    guint num_surfaces_per_frame;

    GstCaps *sink_caps;

    GMutex ctx_lock;
    gboolean isAudio;
};

struct _GstNvStreamDemuxClass {
    GstElementClass parent_class;
};

G_GNUC_INTERNAL GType gst_nvstreamdemux_2_get_type(void);

G_END_DECLS
#endif
