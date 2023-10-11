/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * @file nvds_appctx_server.h
 * <b>NVIDIA DeepStream Yaml REST Server Application context API Specification </b>
 *
 * @b Description: This file specifies details required to parse REST server
 * application context. This would be required if nvds_rest_server is
 * configured to use with the application.
 */

/**
 * @defgroup  REST server_appctx  DeepStream Yaml Parser API
 * Defines an API for the GStreamer REST Application Server context
 * @ingroup custom_gstreamer
 * @{
 */

#ifndef _NVGSTDS_APPCTX_SERVER_PARSER_H_
#define _NVGSTDS_APPCTX_SERVER_PARSER_H_

#ifdef __cplusplus

#include "gst-nvmultiurisrcbincreator.h"
#include "nvds_rest_server.h"
#include "nvds_yml_parser.h"
extern "C" {
#endif

#include <gst/gst.h>

/**
 * REST server application context
 */
typedef struct {
    GstElement *pipeline;
    GstElement *multiuribin;
    GstElement *sink;
    GstElement *pgie;
    GstElement *queue1;
    GstElement *queue2;
    GstElement *queue3;
    GstElement *queue4;
    GstElement *queue5;
    GstElement *nvvidconv;
    GstElement *tiler;
    GstElement *nvdslogger;
    GstElement *preprocess;
    GstElement *nvosd;
    GstElement *nvvidconv2;
    GstElement *encoder;
    GstElement *parser;
    GstElement *queue_post_encoder;

    void *restServer;
    NvDsServerConfig server_conf;
    gchar *httpIp;
    gchar *httpPort;

    GMutex bincreator_lock;
    NvDst_Handle_NvMultiUriSrcCreator nvmultiurisrcbinCreator;
    GstDsNvStreammuxConfig muxConfig;
    GstDsNvUriSrcConfig config;
    guint sourceIdCounter;
    gchar *uri_list;
} AppCtx;

/**
 * Set AppCtx from values specified in a YAML configuration file.
 *
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             application context values to be updated.
 * @param[in]  appctx The context which gets updated as per the parameter key(s)
 *             values defined in the group
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_server_appctx(gchar *cfg_file_path,
                                              const char *group,
                                              AppCtx *appctx);

/**
 * Set "within_multiurisrcbin" varible from values specified in a YAML configuration file.
 *
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             application context values to be updated.
 * @param[in]  within_multiurisrcbin The boolean variable updated as per the parameter key(s)
 *             values defined in the group. If True, nvds_rest_server library is to be used
 *             directly.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_check_rest_server_with_app(gchar *cfg_file_path,
                                                           const char *group,
                                                           gboolean *within_multiurisrcbin);

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_APPCTX_SERVER_PARSER_H_ */

/** @} */
