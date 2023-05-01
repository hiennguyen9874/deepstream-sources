/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _NVDS_SERVER_H_
#define _NVDS_SERVER_H_

#include <functional>
#include <string>
#include <vector>

#include "gst-nvcustomevent.h"

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
} NvDsDecStatus;

typedef enum {
    INFER_INTERVAL_UPDATE_SUCCESS = 0,
    INFER_INTERVAL_UPDATE_FAIL,
} NvDsInferStatus;

typedef struct NvDsDecInfo {
    std::string root_key;
    std::string stream_id;
    guint drop_frame_interval;
    NvDsDecStatus status;
} NvDsDecInfo;

typedef struct NvDsRoiInfo {
    std::string root_key;
    std::string stream_id;
    guint roi_count;
    std::vector<RoiDimension> vect;
    NvDsRoiStatus status;
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
} NvDsStreamInfo;

typedef struct NvDsInferInfo {
    std::string root_key;
    std::string stream_id;
    guint interval;
    NvDsInferStatus status;
} NvDsInferInfo;

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
    std::function<void(NvDsStreamInfo *stream_info, void *ctx)> stream_cb;
    std::function<void(NvDsInferInfo *infer_info, void *ctx)> infer_cb;
} NvDsServerCallbacks;

class NvDsRestServer;
NvDsRestServer *nvds_rest_server_start(NvDsServerConfig *server_config,
                                       NvDsServerCallbacks *server_cb);
void nvds_rest_server_stop(NvDsRestServer *ctx);

#endif
