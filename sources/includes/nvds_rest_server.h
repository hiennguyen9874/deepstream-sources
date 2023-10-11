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

#ifndef _NVDS_SERVER_H_
#define _NVDS_SERVER_H_

#include <functional>
#include <string>
#include <vector>

#include "gst-nvdscustomevent.h"

typedef enum {
    DROP_FRAME_INTERVAL = 1 << 0,
    SKIP_FRAMES = 1 << 1,
    LOW_LATENCY_MODE = 1 << 2,
} NvDsDecPropFlag;

typedef enum {
    BITRATE = 1 << 0,
    FORCE_IDR = 1 << 1,
    FORCE_INTRA = 1 << 2,
    IFRAME_INTERVAL = 1 << 3,
} NvDsEncPropFlag;

typedef enum {
    SRC_CROP = 1 << 0,
    DEST_CROP = 1 << 1,
    FLIP_METHOD = 1 << 2,
    INTERPOLATION_METHOD = 1 << 3,
} NvDsConvPropFlag;

typedef enum {
    BATCHED_PUSH_TIMEOUT = 1 << 0,
    MAX_LATENCY = 1 << 1,
} NvDsMuxPropFlag;

typedef enum {
    INFER_INTERVAL = 1 << 0,
} NvDsInferPropFlag;

typedef enum {
    INFERSERVER_INTERVAL = 1 << 0,
} NvDsInferServerPropFlag;

typedef enum {
    PROCESS_MODE = 1 << 0,
} NvDsOsdPropFlag;

typedef enum {
    ROI_UPDATE = 1 << 0,
} NvDsRoiPropFlag;

typedef enum {
    QUIT_APP = 1 << 0,
} NvDsAppInstanceFlag;

typedef enum {
    QUIT_SUCCESS = 0,
    QUIT_FAIL,
} NvDsAppInstanceStatus;

typedef enum {
    STREAM_ADD_SUCCESS = 0,
    STREAM_ADD_FAIL,
    STREAM_REMOVE_SUCCESS,
    STREAM_REMOVE_FAIL,
} NvDsStreamStatus;

typedef enum {
    ROI_UPDATE_SUCCESS = 0,
    ROI_UPDATE_FAIL,
} NvDsRoiStatus;

typedef enum {
    DROP_FRAME_INTERVAL_UPDATE_SUCCESS = 0,
    DROP_FRAME_INTERVAL_UPDATE_FAIL,
    SKIP_FRAMES_UPDATE_SUCCESS,
    SKIP_FRAMES_UPDATE_FAIL,
    LOW_LATENCY_MODE_UPDATE_SUCCESS,
    LOW_LATENCY_MODE_UPDATE_FAIL,
} NvDsDecStatus;

typedef enum {
    BITRATE_UPDATE_SUCCESS = 0,
    BITRATE_UPDATE_FAIL,
    FORCE_IDR_UPDATE_SUCCESS,
    FORCE_IDR_UPDATE_FAIL,
    FORCE_INTRA_UPDATE_SUCCESS,
    FORCE_INTRA_UPDATE_FAIL,
    IFRAME_INTERVAL_UPDATE_SUCCESS,
    IFRAME_INTERVAL_UPDATE_FAIL,
} NvDsEncStatus;

typedef enum {
    DEST_CROP_UPDATE_SUCCESS = 0,
    DEST_CROP_UPDATE_FAIL,
    SRC_CROP_UPDATE_SUCCESS,
    SRC_CROP_UPDATE_FAIL,
    INTERPOLATION_METHOD_UPDATE_SUCCESS,
    INTERPOLATION_METHOD_UPDATE_FAIL,
    FLIP_METHOD_UPDATE_SUCCESS,
    FLIP_METHOD_UPDATE_FAIL,
} NvDsConvStatus;

typedef enum {
    BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS = 0,
    BATCHED_PUSH_TIMEOUT_UPDATE_FAIL,
    MAX_LATENCY_UPDATE_SUCCESS,
    MAX_LATENCY_UPDATE_FAIL,
} NvDsMuxStatus;

typedef enum {
    INFER_INTERVAL_UPDATE_SUCCESS = 0,
    INFER_INTERVAL_UPDATE_FAIL,
} NvDsInferStatus;

typedef enum {
    INFERSERVER_INTERVAL_UPDATE_SUCCESS = 0,
    INFERSERVER_INTERVAL_UPDATE_FAIL,
} NvDsInferServerStatus;

typedef enum {
    PROCESS_MODE_UPDATE_SUCCESS = 0,
    PROCESS_MODE_UPDATE_FAIL,
} NvDsOsdStatus;

typedef struct NvDsDecInfo {
    std::string root_key;
    std::string stream_id;
    guint drop_frame_interval;
    guint skip_frames;
    gboolean low_latency_mode;
    NvDsDecStatus status;
    NvDsDecPropFlag dec_flag;
    std::string dec_log;
} NvDsDecInfo;

typedef struct NvDsEncInfo {
    std::string root_key;
    std::string stream_id;
    guint bitrate;
    gboolean force_idr;
    gboolean force_intra;
    guint iframeinterval;
    NvDsEncStatus status;
    NvDsEncPropFlag enc_flag;
    std::string enc_log;
} NvDsEncInfo;

typedef struct NvDsConvInfo {
    std::string root_key;
    std::string stream_id;
    std::string src_crop;
    std::string dest_crop;
    guint flip_method;
    guint interpolation_method;
    NvDsConvStatus status;
    NvDsConvPropFlag conv_flag;
    std::string conv_log;
} NvDsConvInfo;

typedef struct NvDsMuxInfo {
    std::string root_key;
    gint batched_push_timeout;
    guint max_latency;
    NvDsMuxStatus status;
    NvDsMuxPropFlag mux_flag;
    std::string mux_log;
} NvDsMuxInfo;

typedef struct NvDsRoiInfo {
    std::string root_key;
    std::string stream_id;
    guint roi_count;
    std::vector<RoiDimension> vect;
    NvDsRoiStatus status;
    NvDsRoiPropFlag roi_flag;
    std::string roi_log;
} NvDsRoiInfo;

typedef struct NvDsStreamInfo {
    std::string key;
    std::string value_camera_id;
    std::string value_camera_name;
    std::string value_camera_url;
    std::string value_change;

    std::string metadata_resolution;
    std::string metadata_codec;
    std::string metadata_framerate;

    std::string headers_source;
    std::string headers_created_at;
    NvDsStreamStatus status;
    std::string stream_log;
} NvDsStreamInfo;

typedef struct NvDsInferInfo {
    std::string root_key;
    std::string stream_id;
    guint interval;
    NvDsInferStatus status;
    NvDsInferPropFlag infer_flag;
    std::string infer_log;
} NvDsInferInfo;

typedef struct NvDsOsdInfo {
    std::string root_key;
    std::string stream_id;
    guint process_mode;
    NvDsOsdStatus status;
    NvDsOsdPropFlag osd_flag;
    std::string osd_log;
} NvDsOsdInfo;

typedef struct NvDsAppInstanceInfo {
    std::string root_key;
    gboolean app_quit;
    NvDsAppInstanceStatus status;
    NvDsAppInstanceFlag appinstance_flag;
    std::string app_log;
} NvDsAppInstanceInfo;

typedef struct NvDsInferServerInfo {
    std::string root_key;
    std::string stream_id;
    guint interval;
    NvDsInferServerStatus status;
    NvDsInferServerPropFlag inferserver_flag;
    std::string inferserver_log;
} NvDsInferServerInfo;

typedef struct NvDsResponseInfo {
    std::string status;
    std::string reason;
} NvDsResponseInfo;

typedef struct NvDsServerConfig {
    std::string ip;
    std::string port;
} NvDsServerConfig;

typedef struct NvDsServerCallbacks {
    std::function<void(NvDsRoiInfo *roi_info, void *ctx)> roi_cb;
    std::function<void(NvDsDecInfo *dec_info, void *ctx)> dec_cb;
    std::function<void(NvDsEncInfo *enc_info, void *ctx)> enc_cb;
    std::function<void(NvDsStreamInfo *stream_info, void *ctx)> stream_cb;
    std::function<void(NvDsInferInfo *infer_info, void *ctx)> infer_cb;
    std::function<void(NvDsConvInfo *conv_info, void *ctx)> conv_cb;
    std::function<void(NvDsMuxInfo *mux_info, void *ctx)> mux_cb;
    std::function<void(NvDsInferServerInfo *inferserver_info, void *ctx)> inferserver_cb;
    std::function<void(NvDsOsdInfo *osd_info, void *ctx)> osd_cb;
    std::function<void(NvDsAppInstanceInfo *appinstance_info, void *ctx)> appinstance_cb;
} NvDsServerCallbacks;
class NvDsRestServer;
NvDsRestServer *nvds_rest_server_start(NvDsServerConfig *server_config,
                                       NvDsServerCallbacks *server_cb);
void nvds_rest_server_stop(NvDsRestServer *ctx);

#endif
