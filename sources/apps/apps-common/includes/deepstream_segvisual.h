/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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

#ifndef __NVGSTDS_SEGVISUAL_H__
#define __NVGSTDS_SEGVISUAL_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    GstElement *bin;
    GstElement *queue;
    GstElement *nvvidconv;
    GstElement *conv_queue;
    GstElement *cap_filter;
    GstElement *nvsegvisual;
} NvDsSegVisualBin;

typedef struct {
    gboolean enable;
    guint gpu_id;
    guint max_batch_size;
    guint width;
    guint height;
    guint nvbuf_memory_type; /* For nvvidconv */
} NvDsSegVisualConfig;

/**
 * Initialize @ref NvDsSegVisualBin. It creates and adds SegVisual and other elements
 * needed for processing to the bin. It also sets properties mentioned
 * in the configuration file under group @ref CONFIG_GROUP_SegVisual
 *
 * @param[in] config pointer to SegVisual @ref NvDsSegVisualConfig parsed from config file.
 * @param[in] bin pointer to @ref NvDsSegVisualBin to be filled.
 *
 * @return true if bin created successfully.
 */
gboolean create_segvisual_bin(NvDsSegVisualConfig *config, NvDsSegVisualBin *bin);

#ifdef __cplusplus
}
#endif

#endif
