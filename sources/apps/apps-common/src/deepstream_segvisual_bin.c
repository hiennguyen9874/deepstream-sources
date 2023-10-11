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

#include "deepstream_common.h"
#include "deepstream_segvisual.h"

#define SEG_OUTPUT_WIDTH 1280
#define SEG_OUTPUT_HEIGHT 720

gboolean create_segvisual_bin(NvDsSegVisualConfig *config, NvDsSegVisualBin *bin)
{
    gboolean ret = FALSE;

    bin->bin = gst_bin_new("segvisual_bin");
    if (!bin->bin) {
        NVGSTDS_ERR_MSG_V("Failed to create 'segvisual_bin'");
        goto done;
    }

    bin->nvvidconv = gst_element_factory_make(NVDS_ELEM_VIDEO_CONV, "segvisual_conv");
    if (!bin->nvvidconv) {
        NVGSTDS_ERR_MSG_V("Failed to create 'segvisual_conv'");
        goto done;
    }

    bin->queue = gst_element_factory_make(NVDS_ELEM_QUEUE, "segvisual_queue");
    if (!bin->queue) {
        NVGSTDS_ERR_MSG_V("Failed to create 'segvisual_queue'");
        goto done;
    }

    bin->conv_queue = gst_element_factory_make(NVDS_ELEM_QUEUE, "segvisual_conv_queue");
    if (!bin->conv_queue) {
        NVGSTDS_ERR_MSG_V("Failed to create 'segvisual_conv_queue'");
        goto done;
    }

    bin->nvsegvisual = gst_element_factory_make(NVDS_ELEM_SEGVISUAL, "nvsegvisual0");
    if (!bin->nvsegvisual) {
        NVGSTDS_ERR_MSG_V("Failed to create 'nvsegvisual0'");
        goto done;
    }

    gst_bin_add_many(GST_BIN(bin->bin), bin->queue, bin->nvvidconv, bin->conv_queue,
                     bin->nvsegvisual, NULL);

    g_object_set(G_OBJECT(bin->nvvidconv), "gpu-id", config->gpu_id, NULL);

    g_object_set(G_OBJECT(bin->nvvidconv), "nvbuf-memory-type", config->nvbuf_memory_type, NULL);

    if (config->width != 0 && config->height != 0) {
        g_object_set(G_OBJECT(bin->nvsegvisual), "width", config->width, "height", config->height,
                     NULL);
    } else {
        g_object_set(G_OBJECT(bin->nvsegvisual), "width", SEG_OUTPUT_WIDTH, "height",
                     SEG_OUTPUT_HEIGHT, NULL);
    }

    g_object_set(G_OBJECT(bin->nvsegvisual), "gpu-id", config->gpu_id, NULL);
    g_object_set(G_OBJECT(bin->nvsegvisual), "batch-size", config->max_batch_size, NULL);

    NVGSTDS_LINK_ELEMENT(bin->queue, bin->nvvidconv);

    NVGSTDS_LINK_ELEMENT(bin->nvvidconv, bin->conv_queue);

    NVGSTDS_LINK_ELEMENT(bin->conv_queue, bin->nvsegvisual);

    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->queue, "sink");

    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->nvsegvisual, "src");

    ret = TRUE;
done:
    if (!ret) {
        NVGSTDS_ERR_MSG_V("%s failed", __func__);
    }
    return ret;
}
