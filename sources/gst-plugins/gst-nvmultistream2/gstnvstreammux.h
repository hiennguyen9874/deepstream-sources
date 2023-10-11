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

#ifndef __GST_NVSTREAMMUX_H__
#define __GST_NVSTREAMMUX_H__

#include <gst/gst.h>
#include <gst/video/video.h>
#include <time.h>

#include "GstNvStreamMuxCtx.h"
#include "cuda_runtime_api.h"
#include "gstnvstreammux_ntp.h"
#include "gstnvstreammuxdebug.h"
#include "gstnvtimesynch.h"
#include "nvstreammux.h"
#include "nvstreammux_debug.h"

G_BEGIN_DECLS
#define GST_TYPE_NVSTREAMMUX (gst_nvstreammux_2_get_type())
#define GST_NVSTREAMMUX(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVSTREAMMUX, GstNvStreamMux))
#define GST_NVSTREAMMUX_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVSTREAMMUX, GstNvStreamMuxClass))
#define GST_IS_NVSTREAMMUX(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVSTREAMMUX))
#define GST_IS_NVSTREAMMUX_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVSTREAMMUX))
typedef struct _GstNvStreamMux GstNvStreamMux;
typedef struct _GstNvStreamMuxClass GstNvStreamMuxClass;
#define NEW_METADATA 1

struct _GstNvStreamMux {
    GstElement element;

    GstPad *srcpad;

    GstSegment segment;

    GstFlowReturn last_flow_ret;

    unsigned int width;

    unsigned int height;

    NvStreamMux *helper;

    bool send_eos;

    bool eos_sent;

    bool all_pads_eos;

    gboolean query_resolution;

    unsigned int batch_size;

    GstVideoInfo out_videoinfo;

    unsigned int num_surfaces_per_frame;

    bool pad_task_created;

    gulong frame_duration_nsec;

    gulong cur_frame_pts;
    GstClockTime pts_offset;

    /** Path to the configuration file for this instance of gst-nvstreammux. */
    gchar *config_file_path;
    /** boolean TRUE if a new config_file_path setting is available */
    gboolean config_file_available;

    /** Boolean indicating if the config parsing was successful. */
    gboolean config_file_parse_successful;

    gboolean module_initialized;

    gboolean segment_sent;

    guint num_sink_pads;

    /** Audio support */
    gboolean isAudio;

    /** muxer context */
    GstNvStreamMuxCtx *muxCtx;

    /** property that specifies if the ntp timestamp is from rtspsrc or system */
    gboolean sys_ts;
    GstNvDsNtpCalculatorMode ntp_calc_mode;
    GHashTable *sink_pad_caps;

    /** whether input buffer synchronization is turned ON/OFF */
    gboolean sync_inputs;

    GstClockTime max_latency;

    GstClockTime prev_outbuf_pts;

    NvTimeSync *synch_buffer;

    GstClockTime peer_latency_min;
    GstClockTime peer_latency_max;
    gboolean peer_latency_live;
    gboolean has_peer_latency;
    GMutex ctx_lock;

    gboolean frame_num_reset_on_eos;
    gboolean frame_num_reset_on_stream_reset;
    /** Application specified frame duration used for NTP timestamp calculaion */
    GstClockTime frame_duration;
    bool pushed_stream_start_once;
    gboolean no_pipeline_eos;

    INvStreammuxDebug *debug_iface;
};

struct _GstNvStreamMuxClass {
    GstElementClass parent_class;
};

G_GNUC_INTERNAL GType gst_nvstreammux_2_get_type(void);

G_END_DECLS
#endif
