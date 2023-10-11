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

#include "gst-nvcustomevent.h"
#include "gst-nvmultiurisrcbincreator.h"
#include "nvds_appctx_server.h"
#include "nvds_rest_server.h"

/* Callback to handle application related REST API requests*/
void s_appinstance_callback_impl(NvDsAppInstanceInfo *appinstance_info, void *ctx);

/* Callback to handle osd related REST API requests*/
void s_osd_callback_impl(NvDsOsdInfo *osd_info, void *ctx);

/* Callback to handle nvstreammux related REST API requests*/
void s_mux_callback_impl(NvDsMuxInfo *mux_info, void *ctx);

/* Callback to handle encoder specific REST API requests*/
void s_enc_callback_impl(NvDsEncInfo *enc_info, void *ctx);

/* Callback to handle encoder specific REST API requests*/
void s_conv_callback_impl(NvDsConvInfo *conv_info, void *ctx);

/* Callback to handle nvinferserver specific REST API requests*/
void s_inferserver_callback_impl(NvDsInferServerInfo *inferserver_info, void *ctx);

/* Callback to handle nvinfer specific REST API requests*/
void s_infer_callback_impl(NvDsInferInfo *infer_info, void *ctx);

/* Callback to handle nvv4l2decoder specific REST API requests*/
void s_dec_callback_impl(NvDsDecInfo *dec_info, void *ctx);

/* Callback to handle nvdspreprocess specific REST API requests*/
void s_roi_callback_impl(NvDsRoiInfo *roi_info, void *ctx);

/* Callback to handle stream add/remove specific REST API requests*/
void s_stream_callback_impl(NvDsStreamInfo *stream_info, void *ctx);