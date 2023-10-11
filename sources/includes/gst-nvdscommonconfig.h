/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
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

#ifndef _GST_NVDSCOMMON_CONFIG_H_
#define _GST_NVDSCOMMON_CONFIG_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <gst/gst.h>

typedef enum { SOURCE_TYPE_AUTO, SOURCE_TYPE_URI, SOURCE_TYPE_RTSP } NvDsUriSrcBinType;

typedef enum {
    DEC_SKIP_FRAMES_TYPE_NONE,
    DEC_SKIP_FRAMES_TYPE_NONREF,
    DEC_SKIP_FRAMES_TYPE_KEY_FRAME_ONLY
} NvDsUriSrcBinDecSkipFrame;

typedef enum { RTP_PROTOCOL_MULTI = 0, RTP_PROTOCOL_TCP = 4 } NvDsUriSrcBinRtpProtocol;

typedef enum { SMART_REC_DISABLE, SMART_REC_CLOUD, SMART_REC_MULTI } NvDsUriSrcBinSRType;

typedef enum {
    SMART_REC_AUDIO_VIDEO,
    SMART_REC_VIDEO_ONLY,
    SMART_REC_AUDIO_ONLY
} NvDsUriSrcBinSRMode;

typedef enum { SMART_REC_MP4, SMART_REC_MKV } NvDsUriSrcBinSRCont;

typedef struct _NvDsSensorInfo {
    guint source_id;
    gchar const *sensor_id;
} NvDsSensorInfo;

typedef struct _GstDsNvUriSrcConfig {
    NvDsUriSrcBinType src_type;
    gboolean loop;
    gchar *uri;
    gint latency;
    NvDsUriSrcBinSRType smart_record;
    gchar *smart_rec_dir_path;
    gchar *smart_rec_file_prefix;
    NvDsUriSrcBinSRCont smart_rec_container;
    NvDsUriSrcBinSRMode smart_rec_mode;
    guint smart_rec_def_duration;
    guint smart_rec_cache_size;
    guint gpu_id;
    gint source_id;
    NvDsUriSrcBinRtpProtocol rtp_protocol;
    guint num_extra_surfaces;
    NvDsUriSrcBinDecSkipFrame skip_frames_type;
    guint cuda_memory_type;
    guint drop_frame_interval;
    gint rtsp_reconnect_interval_sec;
    guint udp_buffer_size;
    gchar *sensorId; /**< unique Sensor ID string */
    gboolean disable_passthrough;
} GstDsNvUriSrcConfig;

typedef struct {
    // Struct members to store config / properties for the element

    // mandatory configs when using legacy nvstreammux
    gint pipeline_width;
    gint pipeline_height;
    gint batched_push_timeout;

    // not mandatory; auto configured
    gint batch_size;

    // not mandatory; defaults will be used
    gint buffer_pool_size;
    gint compute_hw;
    gint num_surfaces_per_frame;
    gint interpolation_method;
    guint gpu_id;
    guint nvbuf_memory_type;
    gboolean live_source;
    gboolean enable_padding;
    gboolean attach_sys_ts_as_ntp;
    gchar *config_file_path;
    gboolean sync_inputs;
    guint64 max_latency;
    gboolean frame_num_reset_on_eos;
    gboolean frame_num_reset_on_stream_reset;

    guint64 frame_duration;
    guint maxBatchSize;
    gboolean async_process;
    gboolean no_pipeline_eos;
} GstDsNvStreammuxConfig;

#ifdef __cplusplus
}
#endif

#endif
