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

#include "gstdsnvmultiurisrcbin.h"

#include <gst/audio/audio.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "gst-nvcommon.h"
#include "gst-nvcustomevent.h"
#include "gst-nvdscustommessage.h"
#include "gst-nvevent.h"
#include "gst-nvmessage.h"
#include "gst-nvquery.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvds_rest_server.h"
#include "nvdsgstutils.h"

// Default prop values
#define DEFAULT_HTTP_IP "localhost"
#define DEFAULT_HTTP_PORT "9000"
#define DEFAULT_MAX_BATCH_SIZE (30)
#define DEFAULT_NUM_EXTRA_SURFACES 1
#define DEFAULT_GPU_DEVICE_ID 0
#define DEFAULT_DROP_FRAME_INTERVAL 0
#define DEFAULT_SOURCE_TYPE 0
#define DEFAULT_DEC_SKIP_FRAME_TYPE 0
#define DEFAULT_RTP_PROTOCOL 0
#define DEFAULT_LATENCY 100
#define DEFAULT_FILE_LOOP FALSE
#define DEFAULT_DISABLE_PASSTHROUGH FALSE
#define DEFAULT_SMART_RECORD_MODE 0
#define DEFAULT_SMART_RECORD_PREFIX "Smart_Record"
#define DEFAULT_SMART_RECORD_CACHE 20
#define DEFAULT_SMART_RECORD_CONTAINER 0
#define DEFAULT_SMART_RECORD_DEFAULT_DURATION 20
#define DEFAULT_RTSP_RECONNECT_INTERVAL 0
#define DEFAULT_SOURCE_ID -1
#define DEFAULT_UDP_BUFFER_SIZE 524288
#define SOURCE_RESET_INTERVAL_SEC 60

/** nvstreammux props: */
#define DEFAULT_BATCH_METHOD 1
#define DEFAULT_BATCH_SIZE 0
#define DEFAULT_BATCHED_PUSH_TIMEOUT -1
#define DEFAULT_WIDTH 0
#define DEFAULT_HEIGHT 0
#define DEFAULT_QUERY_RESOLUTION FALSE
#define DEFAULT_GPU_DEVICE_ID 0
#define DEFAULT_LIVE_SOURCE FALSE
#define DEFAULT_ATTACH_SYS_TIME_STAMP TRUE
#define DEFAULT_ADAPTIVE_BATCH_SIZE FALSE
#define DEFAULT_ASYNC_PROCESS TRUE
#define DEFAULT_NO_PIPELINE_EOS FALSE
#define DEFAULT_FRAME_DURATION (0)
#define DEFAULT_BUFFER_POOL_SIZE (4)
#define MAX_NVBUFFERS 1024
#define MAX_POOL_BUFFERS (MAX_NVBUFFERS)
#define DEFAULT_CONFIG_FILE_PATH NULL

#define GST_TYPE_NVDSURI_SKIP_FRAMES (gst_nvdsurisrc_dec_skip_frames())
#define GST_TYPE_NVDSURI_RTP_PROTOCOL (gst_nvdsurisrc_rtp_protocol())
#define GST_TYPE_NVDSURI_SMART_RECORD_TYPE (gst_nvdsurisrc_smart_record_type())
#define GST_TYPE_NVDSURI_SMART_RECORD_MODE (gst_nvdsurisrc_smart_record_mode())
#define GST_TYPE_NVDSURI_SMART_RECORD_CONTAINER (gst_nvdsurisrc_smart_record_container())

#define GST_TYPE_V4L2_VID_CUDADEC_MEM_TYPE (gst_video_cudadec_mem_type())
#define GST_TYPE_NVMULTIURISRCBIN_MODE (gst_nvmultiurisrcbin_mode())
#define DEFAULT_CUDADEC_MEM_TYPE (0)
#define DEFAULT_NVBUF_MEM_TYPE (0)

typedef struct {
    GstDsNvMultiUriBin *ubin;
    guint sourceId;
    gboolean didSourceElemError;
} DsNvMultiUriBinSourceInfo;
static gpointer GThreadFuncRemoveSource(gpointer data);

static GType gst_nvmultiurisrcbin_mode(void)
{
    static GType qtype = 0;

    if (qtype == 0) {
        static const GEnumValue values[] = {
            {0, "Mode Video-Only",
             "Video streams are muxed together; audio streams are ignored; Default"},
            {1, "Mode Audio-Only", "Audio streams are muxed together; video streams are ignored"},
            {0, NULL, NULL}};

        qtype = g_enum_register_static("GstNvMultiUriSrcBinModeType2", values);
    }
    return qtype;
}

static GType gst_video_cudadec_mem_type(void)
{
    static GType qtype = 0;

    if (qtype == 0) {
        static const GEnumValue values[] = {{0, "Memory type Device", "memtype_device"},
                                            {1, "Memory type Host Pinned", "memtype_pinned"},
                                            {2, "Memory type Unified", "memtype_unified"},
                                            {0, NULL, NULL}};

        qtype = g_enum_register_static("GstNvUriSrcBinCudaDecMemType2", values);
    }
    return qtype;
}

static GType gst_nvdsurisrc_dec_skip_frames(void)
{
    static volatile gsize initialization_value = 0;
    static const GEnumValue skip_type[] = {
        {DEC_SKIP_FRAMES_TYPE_NONE, "Decode all frames", "decode_all"},
        {DEC_SKIP_FRAMES_TYPE_NONREF, "Decode non-ref frames", "decode_non_ref"},
        {DEC_SKIP_FRAMES_TYPE_KEY_FRAME_ONLY, "decode key frames", "decode_key"},
        {0, NULL, NULL}};

    if (g_once_init_enter(&initialization_value)) {
        GType tmp = g_enum_register_static("SkipFrames2", skip_type);
        g_once_init_leave(&initialization_value, tmp);
    }
    return (GType)initialization_value;
}

static GType gst_nvdsurisrc_rtp_protocol(void)
{
    static volatile gsize initialization_value = 0;
    static const GEnumValue rtp_protocol[] = {
        {RTP_PROTOCOL_MULTI, "UDP + UDP Multicast + TCP", "rtp-multi"},
        {RTP_PROTOCOL_TCP, "TCP Only", "rtp-tcp"},
        {0, NULL, NULL}};

    if (g_once_init_enter(&initialization_value)) {
        GType tmp = g_enum_register_static("RtpProtocol2", rtp_protocol);
        g_once_init_leave(&initialization_value, tmp);
    }
    return (GType)initialization_value;
}

static GType gst_nvdsurisrc_smart_record_type(void)
{
    static volatile gsize initialization_value = 0;
    static const GEnumValue smart_rec_type[] = {
        {SMART_REC_DISABLE, "Disable Smart Record", "smart-rec-disable"},
        {SMART_REC_CLOUD, "Trigger Smart Record through cloud messages only", "smart-rec-cloud"},
        {SMART_REC_MULTI, "Trigger Smart Record through cloud and local events", "smart-rec-multi"},
        {0, NULL, NULL}};

    if (g_once_init_enter(&initialization_value)) {
        GType tmp = g_enum_register_static("SmartRecordType2", smart_rec_type);
        g_once_init_leave(&initialization_value, tmp);
    }
    return (GType)initialization_value;
}

static GType gst_nvdsurisrc_smart_record_mode(void)
{
    static volatile gsize initialization_value = 0;
    static const GEnumValue smart_rec_mode[] = {
        {SMART_REC_AUDIO_VIDEO, "Record audio and video if available", "smart-rec-mode-av"},
        {SMART_REC_VIDEO_ONLY, "Record video only if available", "smart-rec-mode-video"},
        {SMART_REC_AUDIO_ONLY, "Record audio only if available", "smart-rec-mode-audio"},
        {0, NULL, NULL}};

    if (g_once_init_enter(&initialization_value)) {
        GType tmp = g_enum_register_static("SmartRecordMode2", smart_rec_mode);
        g_once_init_leave(&initialization_value, tmp);
    }
    return (GType)initialization_value;
}

static GType gst_nvdsurisrc_smart_record_container(void)
{
    static volatile gsize initialization_value = 0;
    static const GEnumValue smart_rec_container[] = {
        {SMART_REC_MP4, "MP4 container", "smart-rec-mp4"},
        {SMART_REC_MKV, "MKV container", "smart-rec-mkv"},
        {0, NULL, NULL}};

    if (g_once_init_enter(&initialization_value)) {
        GType tmp = g_enum_register_static("SmartRecordContainerType2", smart_rec_container);
        g_once_init_leave(&initialization_value, tmp);
    }
    return (GType)initialization_value;
}

#define COMMON_AUDIO_CAPS                  \
    "channels = " GST_AUDIO_CHANNELS_RANGE \
    ", "                                   \
    "rate = (int) [ 1, MAX ]"

static GstStaticPadTemplate gst_nvmultiurisrc_bin_src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
        "memory:NVMM",
        "{ "
        "NV12, RGBA, I420 }") "; "
                              "audio/x-raw(memory:NVMM), "
                              "format = { "
                              "S16LE, F32LE }, "
                              "layout = (string) interleaved, " COMMON_AUDIO_CAPS));

GST_DEBUG_CATEGORY(gst_ds_nvmultiurisrc_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvmultiurisrc_bin_debug

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvmultiurisrc_bin_parent_class parent_class
#define _do_init                                                                   \
    GST_DEBUG_CATEGORY_INIT(gst_ds_nvmultiurisrc_bin_debug, "nvmultiurisrcbin", 0, \
                            "nvmultiurisrcbin element");
G_DEFINE_TYPE_WITH_CODE(GstDsNvMultiUriBin, gst_ds_nvmultiurisrc_bin, GST_TYPE_BIN, _do_init);

static void gst_ds_nvmultiurisrc_bin_set_property(GObject *object,
                                                  guint prop_id,
                                                  const GValue *value,
                                                  GParamSpec *spec);
static void gst_ds_nvmultiurisrc_bin_get_property(GObject *object,
                                                  guint prop_id,
                                                  GValue *value,
                                                  GParamSpec *spec);
static void gst_ds_nvmultiurisrc_bin_finalize(GObject *object);
static GstStateChangeReturn gst_ds_nvmultiurisrc_bin_change_state(GstElement *element,
                                                                  GstStateChange transition);
static void gst_ds_nvmultiurisrc_bin_handle_message(GstBin *bin, GstMessage *message);
static void rest_api_server_start(GstDsNvMultiUriBin *nvmultiurisrcbin);

static void gst_ds_nvmultiurisrc_bin_class_init(GstDsNvMultiUriBinClass *klass)
{
    GObjectClass *gobject_class;
    GstElementClass *gstelement_class;
    GstBinClass *gstbin_class;

    gobject_class = G_OBJECT_CLASS(klass);
    gstelement_class = GST_ELEMENT_CLASS(klass);
    gstbin_class = GST_BIN_CLASS(klass);

    gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_ds_nvmultiurisrc_bin_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_ds_nvmultiurisrc_bin_get_property);
    gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_ds_nvmultiurisrc_bin_finalize);
    gstelement_class->change_state = GST_DEBUG_FUNCPTR(gst_ds_nvmultiurisrc_bin_change_state);
    gstbin_class->handle_message = GST_DEBUG_FUNCPTR(gst_ds_nvmultiurisrc_bin_handle_message);

    gst_element_class_add_static_pad_template(gstelement_class,
                                              &gst_nvmultiurisrc_bin_src_template);
    // gst_element_class_add_static_pad_template (gstelement_class,
    //   &gst_nvmultiurisrc_bin_asrc_template);

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_URI_LIST,
        g_param_spec_string(
            "uri-list", "comma separated URI list of sources", "URI of the file or rtsp source",
            NULL,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_SENSOR_ID_LIST,
        g_param_spec_string(
            "sensor-id-list", "comma separated list of source sensor IDs",
            "this vector is one to one mapped with the uri-list", NULL,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_MODE,
        g_param_spec_enum(
            "mode", "Video-only or Audio-only modes available", "Set Video-only or Audio-only",
            GST_TYPE_NVMULTIURISRCBIN_MODE, NVDS_MULTIURISRCBIN_MODE_VIDEO,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_HTTP_IP,
        g_param_spec_string(
            "ip-address", "Set REST API HTTP IP Address", "REST API HTTP Endpoint", DEFAULT_HTTP_IP,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_DISABLE_PASSTHROUGH,
        g_param_spec_boolean(
            "disable-passthrough", "disable-passthrough",
            "Disable passthrough mode at init time, applicable for nvvideoconvert only.",
            DEFAULT_DISABLE_PASSTHROUGH,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_HTTP_PORT,
        g_param_spec_string(
            "port", "Set REST API HTTP Port number",
            "REST API HTTP Endpoint; Note: User may pass \"0\" to disable REST API Server",
            DEFAULT_HTTP_PORT,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_MAX_BATCH_SIZE,
        g_param_spec_uint(
            "max-batch-size", "Set the maximum batch size to be used for nvstreammux",
            "Maximum number of sources to be supported with this instance of nvmultiurisrcbin", 0,
            G_MAXUINT, DEFAULT_MAX_BATCH_SIZE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    /** @{ For nvurisrcbin  */
    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_NUM_EXTRA_SURF,
        g_param_spec_uint(
            "num-extra-surfaces", "Set extra decoder surfaces",
            "Number of surfaces in addition to minimum decode surfaces given by the decoder", 0,
            G_MAXUINT, DEFAULT_NUM_EXTRA_SURFACES,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_GPU_DEVICE_ID,
        g_param_spec_uint(
            "gpu-id", "Set GPU Device ID", "Set GPU Device ID", 0, G_MAXUINT, DEFAULT_GPU_DEVICE_ID,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_CUDADEC_MEM_TYPE,
        g_param_spec_enum(
            "cudadec-memtype", "Memory type for cuda decoder buffers",
            "Set to specify memory type for cuda decoder buffers",
            GST_TYPE_V4L2_VID_CUDADEC_MEM_TYPE, DEFAULT_CUDADEC_MEM_TYPE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_DROP_FRAME_INTERVAL,
        g_param_spec_uint(
            "drop-frame-interval", "Set decoder drop frame interval",
            "Interval to drop the frames,ex: value of 5 means every 5th frame will be given by "
            "decoder, rest all dropped",
            0, 30, DEFAULT_DROP_FRAME_INTERVAL,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_DEC_SKIP_FRAMES,
        g_param_spec_enum(
            "dec-skip-frames", "Type of frames to skip during decoding",
            "Type of frames to skip during decoding", GST_TYPE_NVDSURI_SKIP_FRAMES,
            DEFAULT_DEC_SKIP_FRAME_TYPE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_RTP_PROTOCOL,
        g_param_spec_enum(
            "select-rtp-protocol", "Transport Protocol to use for RTP",
            "Transport Protocol to use for RTP", GST_TYPE_NVDSURI_RTP_PROTOCOL,
            DEFAULT_RTP_PROTOCOL,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_FILE_LOOP,
        g_param_spec_boolean("file-loop", "Loop file sources after EOS",
                             "Loop file sources after EOS. Src type must be source-type-uri and "
                             "uri starting with 'file:/'",
                             DEFAULT_FILE_LOOP,
                             (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_RTSP_RECONNECT_INTERVAL,
        g_param_spec_uint(
            "rtsp-reconnect-interval", "RTSP Reconnect Interval",
            "Timeout in seconds to wait since last data was received from an RTSP source before "
            "forcing a reconnection. 0=disable timeout",
            0, G_MAXUINT, DEFAULT_RTSP_RECONNECT_INTERVAL,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_LATENCY,
        g_param_spec_uint(
            "latency", "Latency",
            "Jitterbuffer size in milliseconds; applicable only for RTSP streams.", 0, G_MAXUINT,
            DEFAULT_LATENCY,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_UDP_BUFFER_SIZE,
        g_param_spec_uint(
            "udp-buffer-size", "UDP Buffer Size",
            "UDP Buffer Size in bytes; applicable only for RTSP streams.", 0, G_MAXUINT,
            DEFAULT_UDP_BUFFER_SIZE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_SMART_RECORD,
        g_param_spec_enum("smart-record", "Enable Smart Record",
                          "Enable Smart Record and choose the type of events to respond to. "
                          "Sources must be of type source-type-rtsp",
                          GST_TYPE_NVDSURI_SMART_RECORD_TYPE, DEFAULT_SMART_RECORD_MODE,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_SMART_RECORD_DIR_PATH,
        g_param_spec_string(
            "smart-rec-dir-path", "Path of directory to save the recorded file",
            "Path of directory to save the recorded file.", NULL,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_SMART_RECORD_FILE_PREFIX,
        g_param_spec_string(
            "smart-rec-file-prefix", "Prefix of file name for recorded video",
            "By default, Smart_Record is the prefix. For unique file names every source must be "
            "provided with a unique prefix",
            DEFAULT_SMART_RECORD_PREFIX,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_SMART_RECORD_VIDEO_CACHE,
        g_param_spec_uint(
            "smart-rec-video-cache", "Size of video cache in seconds.",
            "Size of video cache in seconds. DEPRECATED: Use 'smart-rec-cache' instead", 0,
            G_MAXUINT, DEFAULT_SMART_RECORD_CACHE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_SMART_RECORD_CACHE,
        g_param_spec_uint(
            "smart-rec-cache", "Size of cache in seconds, applies to both audio and video cache",
            "Size of cache in seconds, applies to both audio and video cache", 0, G_MAXUINT,
            DEFAULT_SMART_RECORD_CACHE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_SMART_RECORD_CONTAINER,
        g_param_spec_enum("smart-rec-container", "Container format of recorded video",
                          "Container format of recorded video. MP4 and MKV containers are "
                          "supported. Sources must be of type source-type-rtsp",
                          GST_TYPE_NVDSURI_SMART_RECORD_CONTAINER, DEFAULT_SMART_RECORD_CONTAINER,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_SMART_RECORD_MODE,
        g_param_spec_enum("smart-rec-mode", "Smart record mode", "Smart record mode",
                          GST_TYPE_NVDSURI_SMART_RECORD_MODE, DEFAULT_SMART_RECORD_MODE,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, MULTIURIBIN_PROP_SMART_RECORD_DEFAULT_DURATION,
        g_param_spec_uint(
            "smart-rec-default-duration",
            "In case a Stop event is not generated. This parameter will ensure the recording is "
            "stopped after a predefined default duration.",
            "In case a Stop event is not generated. This parameter will ensure the recording is "
            "stopped after a predefined default duration.",
            0, G_MAXUINT, DEFAULT_SMART_RECORD_DEFAULT_DURATION,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    /** @} For nvurisrcbin */

    /** @{ For nvstreammux */
    g_object_class_install_property(
        gobject_class, PROP_BATCHED_PUSH_TIMEOUT,
        g_param_spec_int("batched-push-timeout", "(nvstreammux) Batched Push Timeout",
                         "Timeout in microseconds to wait after the first buffer is available\n"
                         "\t\t\tto push the batch even if the complete batch is not formed.\n"
                         "\t\t\tSet to -1 to wait infinitely",
                         -1, G_MAXINT, DEFAULT_BATCHED_PUSH_TIMEOUT,
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_WIDTH,
        g_param_spec_uint(
            "width", "(nvstreammux) Width",
            "Width of each frame in output batched buffer. This property MUST be set.", 0,
            G_MAXUINT, DEFAULT_WIDTH, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_HEIGHT,
        g_param_spec_uint(
            "height", "(nvstreammux) Height",
            "Height of each frame in output batched buffer. This property MUST be set.", 0,
            G_MAXUINT, DEFAULT_HEIGHT, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_ENABLE_PADDING,
        g_param_spec_boolean(
            "enable-padding", "(nvstreammux) Enable Padding",
            "Maintain input aspect ratio when scaling by padding with black bands.", FALSE,
            (GParamFlags)(G_PARAM_READWRITE)));

    g_object_class_install_property(
        gobject_class, PROP_NUM_SURFACES_PER_FRAME,
        g_param_spec_uint("num-surfaces-per-frame",
                          "(nvstreammux) Max number of surfaces per frame",
                          "Max number of surfaces per frame", 1, 4, 1,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_LIVE_SOURCE,
        g_param_spec_boolean("live-source", "(nvstreammux) live source",
                             "Boolean property to inform muxer that sources are live.",
                             DEFAULT_LIVE_SOURCE,
                             (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_SYNC_INPUTS,
        g_param_spec_boolean("sync-inputs", "(nvstreammux) Synchronize Inputs",
                             "Boolean property to force sychronization of input frames.", 0,
                             (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_BUFFER_POOL_SIZE,
        g_param_spec_uint("buffer-pool-size", "Buffer Pool Size",
                          "Maximum number of buffers from muxer's output pool",
                          DEFAULT_BUFFER_POOL_SIZE, MAX_POOL_BUFFERS, DEFAULT_BUFFER_POOL_SIZE,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_MAX_LATNECY,
        g_param_spec_uint("max-latency", "maximum lantency",
                          "Additional latency in live mode to allow upstream to take longer to "
                          "produce buffers for the current position (in nanoseconds)",
                          0, G_MAXUINT, 0 /*200000000 */,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_ATTACH_SYS_TIME_STAMP,
        g_param_spec_boolean(
            "attach-sys-ts", "Set system timestamp as ntp timestamp",
            "If set to TRUE, system timestamp will be attached as ntp timestamp.\n"
            "\t\t\tIf set to FALSE, ntp timestamp from rtspsrc, if available, will be attached.",
            DEFAULT_ATTACH_SYS_TIME_STAMP,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_CONFIG_FILE_PATH,
        g_param_spec_string(
            "config-file-path", "Set config file path",
            "Configuation file path (applicable for new nvstreammux)", DEFAULT_CONFIG_FILE_PATH,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    PROP_NVBUF_MEMORY_TYPE_INSTALL(gobject_class);
    PROP_COMPUTE_HW_INSTALL(gobject_class);

    g_object_class_install_property(
        gobject_class, PROP_INTERPOLATION_METHOD,
        g_param_spec_enum("interpolation-method", "Interpolation-method",
                          "Set interpolation methods", GST_TYPE_INTERPOLATION_METHOD,
                          NvBufSurfTransformInter_Bilinear,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                                        GST_PARAM_CONTROLLABLE | G_PARAM_CONSTRUCT)));

    g_object_class_install_property(
        gobject_class, PROP_FRAME_NUM_RESET_ON_EOS,
        g_param_spec_boolean("frame-num-reset-on-eos", "Frame Number Reset on EOS",
                             "Reset frame numbers to 0 for a source from which EOS is received "
                             "(For debugging purpose only)",
                             FALSE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_FRAME_NUM_RESET_ON_STREAM_RESET,
        g_param_spec_boolean(
            "frame-num-reset-on-stream-reset", "Frame Number Reset on stream reset",
            "Reset frame numbers to 0 for a source which needs to be reset. (For debugging purpose "
            "only)\n"
            "Needs to be paired with tracking-id-reset-mode=1 in the tracker config.",
            FALSE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_FRAME_DURATION,
        g_param_spec_int("frame-duration", "Frame duration",
                         "Duration of input frames in milliseconds for use in NTP timestamp "
                         "correction based on frame rate.\n"
                         "\t\t\tIf set to 0, frame duration is inferred automatically from PTS "
                         "values (default).\n"
                         "\t\t\tIf set to -1, disables frame rate based NTP timestamp correction.",
                         -1, G_MAXINT, DEFAULT_FRAME_DURATION,
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_ASYNC_PROCESS,
        g_param_spec_boolean("async-process", "(nvstreammux) Asynchronous Process",
                             "Boolean property to enable/disable asynchronous processing of input "
                             "frames for performance.",
                             DEFAULT_ASYNC_PROCESS,
                             (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_NO_PIPELINE_EOS,
        g_param_spec_boolean("drop-pipeline-eos", "(nvstreammux) No Pipeline EOS",
                             "Boolean property so that EOS is not propagated downstream when all "
                             "source pads are at EOS.",
                             DEFAULT_NO_PIPELINE_EOS,
                             (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    /** @} */

    gst_element_class_set_details_simple(
        gstelement_class, "NvMultiUri Bin", "NvMultiUri Bin", "Nvidia DeepStreamSDK NvMultiUri Bin",
        "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
        "@ https://devtalk.nvidia.com/default/board/209/");
}

static void gst_ds_nvmultiurisrc_bin_set_property(GObject *object,
                                                  guint prop_id,
                                                  const GValue *value,
                                                  GParamSpec *pspec)
{
    GstDsNvMultiUriBin *nvmultiurisrcbin = GST_DS_NVMULTIURISRC_BIN(object);
    GstDsNvUriSrcConfig *config = nvmultiurisrcbin->config;
    GstDsNvStreammuxConfig *muxConfig = nvmultiurisrcbin->muxConfig;

    switch (prop_id) {
    case MULTIURIBIN_PROP_URI_LIST:
        if (nvmultiurisrcbin->uriList) {
            g_free(nvmultiurisrcbin->uriList);
            nvmultiurisrcbin->uriList = NULL;
        }
        if (nvmultiurisrcbin->uriListV) {
            g_strfreev(nvmultiurisrcbin->uriListV);
            nvmultiurisrcbin->uriListV = NULL;
        }
        nvmultiurisrcbin->uriList = g_value_dup_string(value);
        nvmultiurisrcbin->uriListV = g_strsplit(nvmultiurisrcbin->uriList, ",", -1);
        break;
    case MULTIURIBIN_PROP_SENSOR_ID_LIST:
        if (nvmultiurisrcbin->sensorIdList) {
            g_free(nvmultiurisrcbin->sensorIdList);
            nvmultiurisrcbin->sensorIdList = NULL;
        }
        if (nvmultiurisrcbin->sensorIdListV) {
            g_strfreev(nvmultiurisrcbin->sensorIdListV);
            nvmultiurisrcbin->sensorIdListV = NULL;
        }
        nvmultiurisrcbin->sensorIdList = g_value_dup_string(value);
        nvmultiurisrcbin->sensorIdListV = g_strsplit(nvmultiurisrcbin->sensorIdList, ",", -1);
        break;
    case MULTIURIBIN_PROP_MODE:
        nvmultiurisrcbin->mode = (NvDsMultiUriMode)g_value_get_enum(value);
        break;
    case MULTIURIBIN_PROP_HTTP_IP:
        if (nvmultiurisrcbin->httpIp) {
            g_free(nvmultiurisrcbin->httpIp);
        }
        nvmultiurisrcbin->httpIp = g_value_dup_string(value);
        break;
    case MULTIURIBIN_PROP_HTTP_PORT:
        if (nvmultiurisrcbin->httpPort) {
            g_free(nvmultiurisrcbin->httpPort);
        }
        nvmultiurisrcbin->httpPort = g_value_dup_string(value);
        break;
    case MULTIURIBIN_PROP_MAX_BATCH_SIZE:
        muxConfig->maxBatchSize = g_value_get_uint(value);
        break;
    case MULTIURIBIN_PROP_RTSP_RECONNECT_INTERVAL:
        config->rtsp_reconnect_interval_sec = g_value_get_uint(value);
        break;
    case MULTIURIBIN_PROP_NUM_EXTRA_SURF:
        config->num_extra_surfaces = g_value_get_uint(value);
        break;
    case MULTIURIBIN_PROP_DEC_SKIP_FRAMES:
        config->skip_frames_type = (NvDsUriSrcBinDecSkipFrame)g_value_get_enum(value);
        break;
    case MULTIURIBIN_PROP_GPU_DEVICE_ID:
        config->gpu_id = g_value_get_uint(value);
        muxConfig->gpu_id = g_value_get_uint(value);
        break;
    case MULTIURIBIN_PROP_CUDADEC_MEM_TYPE:
        config->cuda_memory_type = g_value_get_enum(value);
        break;
    case MULTIURIBIN_PROP_DROP_FRAME_INTERVAL:
        config->drop_frame_interval = g_value_get_uint(value);
        break;
    case MULTIURIBIN_PROP_RTP_PROTOCOL:
        config->rtp_protocol = (NvDsUriSrcBinRtpProtocol)g_value_get_enum(value);
        break;
    case MULTIURIBIN_PROP_FILE_LOOP:
        config->loop = g_value_get_boolean(value);
        break;
    case MULTIURIBIN_PROP_LATENCY:
        config->latency = g_value_get_uint(value);
        break;
    case MULTIURIBIN_PROP_UDP_BUFFER_SIZE:
        config->udp_buffer_size = g_value_get_uint(value);
        break;
    case MULTIURIBIN_PROP_SMART_RECORD:
        config->smart_record = (NvDsUriSrcBinSRType)g_value_get_enum(value);
        break;
    case MULTIURIBIN_PROP_SMART_RECORD_DIR_PATH:
        config->smart_rec_dir_path = g_value_dup_string(value);
        break;
    case MULTIURIBIN_PROP_SMART_RECORD_FILE_PREFIX:
        config->smart_rec_file_prefix = g_value_dup_string(value);
        break;
    case MULTIURIBIN_PROP_SMART_RECORD_VIDEO_CACHE:
        g_warning(
            "%s: Deprecated property 'smart-rec-video-cache' set. Set property 'smart-rec-cache' "
            "instead.",
            GST_ELEMENT_NAME(nvmultiurisrcbin));
    case MULTIURIBIN_PROP_SMART_RECORD_CACHE:
        config->smart_rec_cache_size = g_value_get_uint(value);
        break;
    case MULTIURIBIN_PROP_SMART_RECORD_CONTAINER:
        config->smart_rec_container = (NvDsUriSrcBinSRCont)g_value_get_enum(value);
        break;
    case MULTIURIBIN_PROP_SMART_RECORD_MODE:
        config->smart_rec_mode = (NvDsUriSrcBinSRMode)g_value_get_enum(value);
        break;
    case MULTIURIBIN_PROP_SMART_RECORD_DEFAULT_DURATION:
        config->smart_rec_def_duration = g_value_get_uint(value);
        break;
    case MULTIURIBIN_PROP_DISABLE_PASSTHROUGH:
        config->disable_passthrough = g_value_get_boolean(value);
        break;
    case PROP_BATCH_SIZE:
        muxConfig->batch_size = g_value_get_uint(value);
        break;
    case PROP_BATCHED_PUSH_TIMEOUT:
        muxConfig->batched_push_timeout = g_value_get_int(value);
        break;
    case PROP_WIDTH:
        muxConfig->pipeline_width = g_value_get_uint(value);
        break;
    case PROP_HEIGHT:
        muxConfig->pipeline_height = g_value_get_uint(value);
        break;
    case PROP_NUM_SURFACES_PER_FRAME:
        muxConfig->num_surfaces_per_frame = g_value_get_uint(value);
        break;
    case PROP_LIVE_SOURCE:
        muxConfig->live_source = g_value_get_boolean(value);
        break;
    case PROP_SYNC_INPUTS:
        muxConfig->sync_inputs = g_value_get_boolean(value);
        break;
    case PROP_ATTACH_SYS_TIME_STAMP:
        muxConfig->attach_sys_ts_as_ntp = g_value_get_boolean(value);
        break;
    case PROP_CONFIG_FILE_PATH:
        muxConfig->config_file_path = g_value_dup_string(value);
        break;
    case PROP_ENABLE_PADDING:
        muxConfig->enable_padding = g_value_get_boolean(value);
        break;
    case PROP_COMPUTE_HW:
        muxConfig->compute_hw = g_value_get_enum(value);
        break;
    case PROP_NVBUF_MEMORY_TYPE:
        muxConfig->nvbuf_memory_type = g_value_get_enum(value);
        break;
    case PROP_INTERPOLATION_METHOD:
        muxConfig->interpolation_method = g_value_get_enum(value);
        break;
    case PROP_BUFFER_POOL_SIZE:
        muxConfig->buffer_pool_size = g_value_get_uint(value);
        break;
    case PROP_MAX_LATNECY:
        muxConfig->max_latency = g_value_get_uint(value);
        break;
    case PROP_FRAME_NUM_RESET_ON_EOS:
        muxConfig->frame_num_reset_on_eos = g_value_get_boolean(value);
        break;
    case PROP_FRAME_NUM_RESET_ON_STREAM_RESET:
        muxConfig->frame_num_reset_on_stream_reset = g_value_get_boolean(value);
        break;
    case PROP_ASYNC_PROCESS:
        muxConfig->async_process = g_value_get_boolean(value);
        break;
    case PROP_NO_PIPELINE_EOS:
        muxConfig->no_pipeline_eos = g_value_get_boolean(value);
        break;
    case PROP_FRAME_DURATION: {
        gint ms_value = g_value_get_int(value);
        if (ms_value < 0) {
            muxConfig->frame_duration = GST_CLOCK_TIME_NONE;
        } else {
            muxConfig->frame_duration = (GstClockTime)ms_value * GST_MSECOND;
        }
        break;
    }
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_ds_nvmultiurisrc_bin_get_property(GObject *object,
                                                  guint prop_id,
                                                  GValue *value,
                                                  GParamSpec *pspec)
{
    GstDsNvMultiUriBin *nvmultiurisrcbin = GST_DS_NVMULTIURISRC_BIN(object);
    GstDsNvUriSrcConfig *config = nvmultiurisrcbin->config;
    GstDsNvStreammuxConfig *muxConfig = nvmultiurisrcbin->muxConfig;

    switch (prop_id) {
    case MULTIURIBIN_PROP_URI_LIST:
        g_value_set_string(value, nvmultiurisrcbin->uriList);
        break;
    case MULTIURIBIN_PROP_SENSOR_ID_LIST:
        g_value_set_string(value, nvmultiurisrcbin->sensorIdList);
        break;
    case MULTIURIBIN_PROP_MODE:
        g_value_set_enum(value, (guint)nvmultiurisrcbin->mode);
        break;
    case MULTIURIBIN_PROP_HTTP_IP:
        g_value_set_string(value, nvmultiurisrcbin->httpIp);
        break;
    case MULTIURIBIN_PROP_HTTP_PORT:
        g_value_set_string(value, nvmultiurisrcbin->httpPort);
        break;
    case MULTIURIBIN_PROP_MAX_BATCH_SIZE:
        g_value_set_uint(value, muxConfig->maxBatchSize);
        break;
    case MULTIURIBIN_PROP_NUM_EXTRA_SURF:
        g_value_set_uint(value, config->num_extra_surfaces);
        break;
    case MULTIURIBIN_PROP_DEC_SKIP_FRAMES:
        g_value_set_enum(value, config->skip_frames_type);
        break;
    case MULTIURIBIN_PROP_GPU_DEVICE_ID:
        // Note: config->gpu_id will be same as muxConfig->gpu_id
        g_value_set_uint(value, config->gpu_id);
        break;
    case MULTIURIBIN_PROP_CUDADEC_MEM_TYPE:
        g_value_set_enum(value, config->cuda_memory_type);
        break;
    case MULTIURIBIN_PROP_DROP_FRAME_INTERVAL:
        g_value_set_uint(value, config->drop_frame_interval);
        break;
    case MULTIURIBIN_PROP_RTP_PROTOCOL:
        g_value_set_enum(value, config->rtp_protocol);
        break;
    case MULTIURIBIN_PROP_FILE_LOOP:
        g_value_set_boolean(value, config->loop);
        break;
    case MULTIURIBIN_PROP_RTSP_RECONNECT_INTERVAL:
        g_value_set_uint(value, config->rtsp_reconnect_interval_sec);
        break;
    case MULTIURIBIN_PROP_LATENCY:
        g_value_set_uint(value, config->latency);
        break;
    case MULTIURIBIN_PROP_UDP_BUFFER_SIZE:
        g_value_set_uint(value, config->udp_buffer_size);
        break;
    case MULTIURIBIN_PROP_SMART_RECORD:
        g_value_set_enum(value, config->smart_record);
        break;
    case MULTIURIBIN_PROP_SMART_RECORD_DIR_PATH:
        g_value_set_string(value, config->smart_rec_dir_path);
        break;
    case MULTIURIBIN_PROP_SMART_RECORD_FILE_PREFIX:
        g_value_set_string(value, config->smart_rec_file_prefix);
        break;
    case MULTIURIBIN_PROP_SMART_RECORD_VIDEO_CACHE:
    case MULTIURIBIN_PROP_SMART_RECORD_CACHE:
        g_value_set_uint(value, config->smart_rec_cache_size);
        break;
    case MULTIURIBIN_PROP_SMART_RECORD_CONTAINER:
        g_value_set_enum(value, config->smart_rec_container);
        break;
    case MULTIURIBIN_PROP_SMART_RECORD_MODE:
        g_value_set_enum(value, config->smart_rec_mode);
        break;
    case MULTIURIBIN_PROP_SMART_RECORD_DEFAULT_DURATION:
        g_value_set_uint(value, config->smart_rec_def_duration);
        break;
    case MULTIURIBIN_PROP_DISABLE_PASSTHROUGH:
        g_value_set_boolean(value, config->disable_passthrough);
        break;
    case PROP_BATCH_SIZE:
        g_value_set_uint(value, muxConfig->batch_size);
        break;
    case PROP_BATCHED_PUSH_TIMEOUT:
        g_value_set_int(value, muxConfig->batched_push_timeout);
        break;
    case PROP_WIDTH:
        g_value_set_uint(value, muxConfig->pipeline_width);
        break;
    case PROP_HEIGHT:
        g_value_set_uint(value, muxConfig->pipeline_height);
        break;
    case PROP_NUM_SURFACES_PER_FRAME:
        g_value_set_uint(value, muxConfig->num_surfaces_per_frame);
        break;
    case PROP_ENABLE_PADDING:
        g_value_set_boolean(value, muxConfig->enable_padding);
        break;
    case PROP_LIVE_SOURCE:
        g_value_set_boolean(value, muxConfig->live_source);
        break;
    case PROP_SYNC_INPUTS:
        g_value_set_boolean(value, muxConfig->sync_inputs);
        break;
    case PROP_ATTACH_SYS_TIME_STAMP:
        g_value_set_boolean(value, muxConfig->attach_sys_ts_as_ntp);
        break;
    case PROP_CONFIG_FILE_PATH:
        g_value_set_string(value, muxConfig->config_file_path);
        break;
    case PROP_COMPUTE_HW:
        g_value_set_enum(value, muxConfig->compute_hw);
        break;
    case PROP_NVBUF_MEMORY_TYPE:
        g_value_set_enum(value, muxConfig->nvbuf_memory_type);
        break;
    case PROP_INTERPOLATION_METHOD:
        g_value_set_enum(value, muxConfig->interpolation_method);
        break;
    case PROP_BUFFER_POOL_SIZE:
        g_value_set_uint(value, muxConfig->buffer_pool_size);
        break;
    case PROP_MAX_LATNECY:
        g_value_set_uint(value, muxConfig->max_latency);
        break;
    case PROP_FRAME_NUM_RESET_ON_EOS:
        g_value_set_boolean(value, muxConfig->frame_num_reset_on_eos);
        break;
    case PROP_FRAME_NUM_RESET_ON_STREAM_RESET:
        g_value_set_boolean(value, muxConfig->frame_num_reset_on_stream_reset);
        break;
    case PROP_FRAME_DURATION: {
        gint ms_value = -1;
        if (muxConfig->frame_duration >= 0) {
            ms_value = muxConfig->frame_duration / GST_MSECOND;
        }
        g_value_set_int(value, ms_value);
        break;
    }
    case PROP_ASYNC_PROCESS:
        g_value_set_boolean(value, muxConfig->async_process);
        break;
    case PROP_NO_PIPELINE_EOS:
        g_value_set_boolean(value, muxConfig->no_pipeline_eos);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_ds_nvmultiurisrc_bin_init(GstDsNvMultiUriBin *nvmultiurisrcbin)
{
    g_object_set(G_OBJECT(nvmultiurisrcbin), "async-handling", TRUE, nullptr);
    nvmultiurisrcbin->config = (GstDsNvUriSrcConfig *)g_malloc0(sizeof(GstDsNvUriSrcConfig));
    nvmultiurisrcbin->muxConfig =
        (GstDsNvStreammuxConfig *)g_malloc0(sizeof(GstDsNvStreammuxConfig));

    nvmultiurisrcbin->muxConfig->nvbuf_memory_type = DEFAULT_NVBUF_MEM_TYPE;
    /** Note: Using max-batch-size config to workaround
     * the pipeline slow-down caused by add/remove without this
     * Issue tracked for next release
     */
    nvmultiurisrcbin->muxConfig->maxBatchSize = DEFAULT_MAX_BATCH_SIZE;
    nvmultiurisrcbin->muxConfig->async_process = DEFAULT_ASYNC_PROCESS;
    nvmultiurisrcbin->muxConfig->no_pipeline_eos = DEFAULT_NO_PIPELINE_EOS;
    nvmultiurisrcbin->muxConfig->attach_sys_ts_as_ntp = DEFAULT_ATTACH_SYS_TIME_STAMP;
    nvmultiurisrcbin->muxConfig->config_file_path = g_strdup(DEFAULT_CONFIG_FILE_PATH);

    nvmultiurisrcbin->config->uri = NULL;
    nvmultiurisrcbin->config->num_extra_surfaces = DEFAULT_NUM_EXTRA_SURFACES;
    nvmultiurisrcbin->config->gpu_id = DEFAULT_GPU_DEVICE_ID;
    nvmultiurisrcbin->config->cuda_memory_type = DEFAULT_CUDADEC_MEM_TYPE;
    nvmultiurisrcbin->config->drop_frame_interval = DEFAULT_DROP_FRAME_INTERVAL;
    nvmultiurisrcbin->config->skip_frames_type =
        (NvDsUriSrcBinDecSkipFrame)DEFAULT_DEC_SKIP_FRAME_TYPE;
    nvmultiurisrcbin->config->rtp_protocol = (NvDsUriSrcBinRtpProtocol)DEFAULT_RTP_PROTOCOL;
    nvmultiurisrcbin->config->rtsp_reconnect_interval_sec = DEFAULT_RTSP_RECONNECT_INTERVAL;
    nvmultiurisrcbin->config->latency = DEFAULT_LATENCY;
    nvmultiurisrcbin->config->disable_passthrough = DEFAULT_DISABLE_PASSTHROUGH;
    nvmultiurisrcbin->config->udp_buffer_size = DEFAULT_UDP_BUFFER_SIZE;
    nvmultiurisrcbin->mode = NVDS_MULTIURISRCBIN_MODE_VIDEO;
    nvmultiurisrcbin->uriList = NULL;
    nvmultiurisrcbin->uriListV = NULL;
    nvmultiurisrcbin->sensorIdList = NULL;
    nvmultiurisrcbin->sensorIdListV = NULL;
    nvmultiurisrcbin->sourceIdCounter = 0;
    nvmultiurisrcbin->bin_src_pad = gst_ghost_pad_new_no_target_from_template(
        "src", gst_static_pad_template_get(&gst_nvmultiurisrc_bin_src_template));
    gst_element_add_pad(GST_ELEMENT(nvmultiurisrcbin), nvmultiurisrcbin->bin_src_pad);
    nvmultiurisrcbin->restServer = NULL;
    nvmultiurisrcbin->httpIp = g_strdup(DEFAULT_HTTP_IP);
    nvmultiurisrcbin->httpPort = g_strdup(DEFAULT_HTTP_PORT);
    g_mutex_init(&nvmultiurisrcbin->bin_lock);

    GST_OBJECT_FLAG_SET(nvmultiurisrcbin, GST_ELEMENT_FLAG_SOURCE);
}

/* Free resources allocated during init. */
static void gst_ds_nvmultiurisrc_bin_finalize(GObject *object)
{
    GstDsNvMultiUriBin *nvmultiurisrcbin = GST_DS_NVMULTIURISRC_BIN(object);

    g_free(nvmultiurisrcbin->config);

    if (nvmultiurisrcbin->uriList) {
        g_free(nvmultiurisrcbin->uriList);
        nvmultiurisrcbin->uriList = NULL;
    }
    if (nvmultiurisrcbin->uriListV) {
        g_strfreev(nvmultiurisrcbin->uriListV);
        nvmultiurisrcbin->uriListV = NULL;
    }

    if (nvmultiurisrcbin->sensorIdList) {
        g_free(nvmultiurisrcbin->sensorIdList);
        nvmultiurisrcbin->sensorIdList = NULL;
    }
    if (nvmultiurisrcbin->sensorIdListV) {
        g_strfreev(nvmultiurisrcbin->sensorIdListV);
        nvmultiurisrcbin->sensorIdListV = NULL;
    }

    if (nvmultiurisrcbin->httpIp) {
        g_free(nvmultiurisrcbin->httpIp);
    }

    if (nvmultiurisrcbin->httpPort) {
        g_free(nvmultiurisrcbin->httpPort);
    }

    if (nvmultiurisrcbin->muxConfig->config_file_path) {
        g_free(nvmultiurisrcbin->muxConfig->config_file_path);
    }

    G_OBJECT_CLASS(parent_class)->finalize(object);
}

static GstStateChangeReturn gst_ds_nvmultiurisrc_bin_change_state(GstElement *element,
                                                                  GstStateChange transition)
{
    GstDsNvMultiUriBin *nvmultiurisrcbin = GST_DS_NVMULTIURISRC_BIN(element);
    GstStateChangeReturn ret;

    if (transition == GST_STATE_CHANGE_NULL_TO_READY) {
        GST_DEBUG_OBJECT(nvmultiurisrcbin, "GST_STATE_CHANGE_NULL_TO_READY %s %d\n", __func__,
                         __LINE__);
        gboolean is_rtsp = nvmultiurisrcbin->config->uri &&
                           g_str_has_prefix(nvmultiurisrcbin->config->uri, "rtsp://");

        (void)is_rtsp;
        guint binNameForCreatorLen = strlen(GST_ELEMENT_NAME((GstElement *)nvmultiurisrcbin));
        gchar *binNameForCreator =
            g_strndup(GST_ELEMENT_NAME((GstElement *)nvmultiurisrcbin), binNameForCreatorLen + 10);
        g_strlcat(binNameForCreator, (const gchar *)"_creator", binNameForCreatorLen + 10);

        /** Initialize the Bin Manipulation API for stream add/remove */
        nvmultiurisrcbin->nvmultiurisrcbinCreator = gst_nvmultiurisrcbincreator_init(
            binNameForCreator, nvmultiurisrcbin->mode, nvmultiurisrcbin->muxConfig);
        if (!nvmultiurisrcbin->nvmultiurisrcbinCreator) {
            GST_ERROR_OBJECT(nvmultiurisrcbin, "Failed to create nvmultiurisrcbin handler\n");
            return GST_STATE_CHANGE_FAILURE;
        }

        gst_bin_add(GST_BIN(nvmultiurisrcbin),
                    gst_nvmultiurisrcbincreator_get_bin(nvmultiurisrcbin->nvmultiurisrcbinCreator));

        /** Add the initial list of sources from uri-list */
        if (nvmultiurisrcbin->uriListV) {
            guint i = 0;
            guint sensorIdListVLen = nvmultiurisrcbin->sensorIdListV
                                         ? g_strv_length(nvmultiurisrcbin->sensorIdListV)
                                         : 0;
            /** Add the initial sources from uri-list */
            for (i = 0; nvmultiurisrcbin->uriListV[i] != NULL; i++) {
                /** Add the source */
                nvmultiurisrcbin->config->uri = nvmultiurisrcbin->uriListV[i];
                if (i < sensorIdListVLen && nvmultiurisrcbin->sensorIdListV[i]) {
                    nvmultiurisrcbin->config->sensorId = nvmultiurisrcbin->sensorIdListV[i];
                }
                nvmultiurisrcbin->config->source_id = 0;
                g_mutex_lock(&nvmultiurisrcbin->bin_lock);
                gboolean ret = gst_nvmultiurisrcbincreator_add_source(
                    nvmultiurisrcbin->nvmultiurisrcbinCreator, nvmultiurisrcbin->config);
                g_mutex_unlock(&nvmultiurisrcbin->bin_lock);
                if (ret == FALSE) {
                    GST_WARNING_OBJECT(nvmultiurisrcbin, "Failed to add uri [%s]\n",
                                       nvmultiurisrcbin->uriListV[i]);
                } else {
                    GST_DEBUG_OBJECT(nvmultiurisrcbin, "Successfully added uri [%s]\n",
                                     nvmultiurisrcbin->uriListV[i]);
                }
                /** clean the config place-holders that we change for each source */
                nvmultiurisrcbin->config->uri = NULL;
                nvmultiurisrcbin->config->sensorId = NULL;
            }
            gst_nvmultiurisrcbincreator_sync_children_states(
                nvmultiurisrcbin->nvmultiurisrcbinCreator);
            gst_element_call_async(GST_ELEMENT(nvmultiurisrcbin),
                                   (GstElementCallAsyncFunc)gst_bin_sync_children_states, NULL,
                                   NULL);
        }

        /** Link the internal bin's proxy src pad with the outer bin's src pad
         * Note: This is done even if uri-list was NULL
         * to allow streams to be added later runtime
         */
        {
            gboolean ret = gst_ghost_pad_set_target(GST_GHOST_PAD(nvmultiurisrcbin->bin_src_pad),
                                                    gst_nvmultiurisrcbincreator_get_source_pad(
                                                        nvmultiurisrcbin->nvmultiurisrcbinCreator));
            if (ret == FALSE) {
                GST_WARNING_OBJECT(nvmultiurisrcbin,
                                   "Failed to set nvstreammux src pad as bin src pad\n");
            } else {
                GST_DEBUG_OBJECT(nvmultiurisrcbin,
                                 "Successfully set nvstreammux src pad as bin src pad\n");
            }
        }

        /** Initialize the HTTP server for REST API */
        if (g_strcmp0(nvmultiurisrcbin->httpPort, "0") != 0) {
            // Valid HTTP port passed; start REST API server
            rest_api_server_start(nvmultiurisrcbin);
        } else {
            GST_WARNING_OBJECT(
                nvmultiurisrcbin,
                "Not starting REST Server as port was passed \"0\"; "
                "Users may still have all sources mentioned with uri-list in the pipeline\n");
        }
    }

    if (transition == GST_STATE_CHANGE_PLAYING_TO_PAUSED) {
        GST_DEBUG_OBJECT(nvmultiurisrcbin, "GST_STATE_CHANGE_PLAYING_TO_PAUSED %s %d\n", __func__,
                         __LINE__);
    }
    if (transition == GST_STATE_CHANGE_READY_TO_PAUSED) {
        GST_DEBUG_OBJECT(nvmultiurisrcbin, "GST_STATE_CHANGE_READY_TO_PAUSED %s %d\n", __func__,
                         __LINE__);
    }
    if (transition == GST_STATE_CHANGE_PAUSED_TO_PLAYING) {
        GST_DEBUG_OBJECT(nvmultiurisrcbin, "GST_STATE_CHANGE_PAUSED_TO_PLAYING %s %d\n", __func__,
                         __LINE__);
    }
    if (transition == GST_STATE_CHANGE_PAUSED_TO_READY) {
        GST_DEBUG_OBJECT(nvmultiurisrcbin, "GST_STATE_CHANGE_PAUSED_TO_READY %s %d\n", __func__,
                         __LINE__);
    }
    if (transition == GST_STATE_CHANGE_READY_TO_NULL) {
        GST_DEBUG_OBJECT(nvmultiurisrcbin, "GST_STATE_CHANGE_READY_TO_NULL %s %d\n", __func__,
                         __LINE__);
        if (nvmultiurisrcbin->nvmultiurisrcbinCreator) {
            gst_nvmultiurisrcbincreator_deinit(nvmultiurisrcbin->nvmultiurisrcbinCreator);
            nvmultiurisrcbin->nvmultiurisrcbinCreator = NULL;
        }
        if (nvmultiurisrcbin->restServer) {
            nvds_rest_server_stop((NvDsRestServer *)nvmultiurisrcbin->restServer);
            nvmultiurisrcbin->restServer = NULL;
        }
    }
    ret = GST_ELEMENT_CLASS(parent_class)->change_state(element, transition);
    return ret;
}

gpointer GThreadFuncRemoveSource(gpointer data)
{
    DsNvMultiUriBinSourceInfo *sourceInfo = (DsNvMultiUriBinSourceInfo *)data;
    GstDsNvMultiUriBin *ubin = sourceInfo->ubin;
    gboolean ret;

    if (sourceInfo->didSourceElemError) {
        /** When source elem throw GStreamer error;
         * forcing state change result in undefined behavior
         * Note: This could also be the second error for the same stream
         * If so, the second call to gst_nvmultiurisrcbincreator_remove_source*
         * will FAIL.
         */
        ret = gst_nvmultiurisrcbincreator_remove_source_without_forced_state_change(
            ubin->nvmultiurisrcbinCreator, sourceInfo->sourceId);
        if (ret == FALSE) {
            GST_WARNING_OBJECT(
                ubin, "Failed to remove sensor; the sensor might have gotten removed already\n");
        } else {
            GST_DEBUG_OBJECT(ubin, "Successfully removed sensor\n");
        }
    } else {
        ret = gst_nvmultiurisrcbincreator_remove_source(ubin->nvmultiurisrcbinCreator,
                                                        sourceInfo->sourceId);
        if (ret == FALSE) {
            GST_WARNING_OBJECT(
                ubin, "Failed to remove sensor; the sensor might have gotten removed already\n");
        } else {
            GST_DEBUG_OBJECT(ubin, "Successfully removed sensor\n");
        }
    }
    g_free(sourceInfo);
    return NULL;
}

static void gst_ds_nvmultiurisrc_bin_handle_message(GstBin *bin, GstMessage *message)
{
    GstDsNvMultiUriBin *ubin = (GstDsNvMultiUriBin *)bin;
    (void)ubin;

    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
        gchar *debug;
        GError *error;

        gst_message_parse_error(message, &error, &debug);
        g_printerr("nvmultiurisrcbin ERROR from element %s: %s\n", GST_OBJECT_NAME(message->src),
                   error->message);

        gboolean didSourceElemError = FALSE;
        if (g_strcmp0(GST_OBJECT_NAME(message->src), "source") == 0 &&
            g_strrstr(error->message, "GStreamer error:")) {
            didSourceElemError = TRUE;
        }
        if (didSourceElemError || g_strrstr(error->message, "Invalid URI") ||
            g_strrstr(error->message, "No URI handler implemented for") ||
            g_strrstr(error->message, "Resource not found") ||
            g_strrstr(error->message, "Could not open resource for")) {
            GST_DEBUG_OBJECT(ubin,
                             "Converting ERROR to WARNING; We need to be able to continue adding "
                             "valid streams\n");
            GST_MESSAGE_TYPE(message) =
                GST_MESSAGE_WARNING; // Change error to warning, so app can continue running

            // Retrieve the nvurisrcbin corresponding to the source so that we may get the source id
            // and remove the broken stream
            GstDsNvUriSrcBin *nvurisrcbin = NULL;
            // If the source of the message is named "source", this is the filesrc of the
            // uridecodebin of the nvurisrcbin. So, the nvurisrcbin will be the src->parent->parent
            if (g_strcmp0(GST_OBJECT_NAME(message->src), "source") == 0) {
                nvurisrcbin = (GstDsNvUriSrcBin *)message->src->parent->parent;
            }
            // "nvurisrc_bin_src_elem" is the uridecodebin of the nvurisrcbin. "src" is the rtspsrc
            // of the nvurisrcbin. In either case, the nvurisrcbin will be the src->parent
            else if (g_strcmp0(GST_OBJECT_NAME(message->src), "nvurisrc_bin_src_elem") == 0 ||
                     g_strcmp0(GST_OBJECT_NAME(message->src), "src") == 0) {
                nvurisrcbin = (GstDsNvUriSrcBin *)message->src->parent;
            }
            // Remove the stream
            if (nvurisrcbin != NULL && GST_IS_OBJECT(nvurisrcbin) && nvurisrcbin->config) {
                g_mutex_lock(&ubin->bin_lock);
                GST_DEBUG_OBJECT(ubin, "removing source %d\n", nvurisrcbin->config->source_id);
                {
                    DsNvMultiUriBinSourceInfo *sourceInfo =
                        (DsNvMultiUriBinSourceInfo *)g_malloc0(sizeof(DsNvMultiUriBinSourceInfo));
                    sourceInfo->ubin = ubin;
                    sourceInfo->sourceId = nvurisrcbin->config->source_id;
                    sourceInfo->didSourceElemError = didSourceElemError;
                    /** Remove the stream from a separate short lived thread */
                    g_thread_new(NULL, GThreadFuncRemoveSource, sourceInfo);
                }
                g_mutex_unlock(&ubin->bin_lock);
            }
        }
        g_error_free(error);
        g_free(debug);
    }

    GST_BIN_CLASS(parent_class)->handle_message(bin, message);
}

static void s_stream_api_impl(NvDsStreamInfo *stream_info, void *ctx)
{
    GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *)ctx;
    (void)nvmultiurisrcbin;

    g_mutex_lock(&nvmultiurisrcbin->bin_lock);

    /** check stream_info->value_change to identify stream add/remove */
    if (g_strrstr(stream_info->value_change.c_str(), "add") ||
        g_strrstr(stream_info->value_change.c_str(), "streaming")) {
        GstDsNvUriSrcConfig **sourceConfigs = NULL;
        guint numSourceConfigs = 0;
        /** Check if we can accomodate more sources */
        if (gst_nvmultiurisrcbincreator_get_active_sources_list(
                nvmultiurisrcbin->nvmultiurisrcbinCreator, &numSourceConfigs, &sourceConfigs)) {
            gst_nvmultiurisrcbincreator_src_config_list_free(
                nvmultiurisrcbin->nvmultiurisrcbinCreator, numSourceConfigs, sourceConfigs);
            if (numSourceConfigs >= nvmultiurisrcbin->muxConfig->maxBatchSize) {
                GST_WARNING_OBJECT(
                    nvmultiurisrcbin,
                    "Failed to add sensor id=[%s]; "
                    "We have [%d] active sources and max-batch-size is configured to [%d]\n",
                    stream_info->value_camera_id.c_str(), numSourceConfigs,
                    nvmultiurisrcbin->muxConfig->maxBatchSize);
                stream_info->status = STREAM_ADD_FAIL;
                g_mutex_unlock(&nvmultiurisrcbin->bin_lock);
                return;
            }
        }

        /** Check if sensor id already exist */
        GstDsNvUriSrcConfig *sourceConfig = NULL;
        if ((sourceConfig = gst_nvmultiurisrcbincreator_get_source_config_by_sensorid(
                 nvmultiurisrcbin->nvmultiurisrcbinCreator,
                 stream_info->value_camera_id.c_str()))) {
            GST_WARNING_OBJECT(nvmultiurisrcbin, "Failed to add sensor id=[%s]; Already added\n",
                               stream_info->value_camera_id.c_str());
            stream_info->status = STREAM_ADD_FAIL;
            g_mutex_unlock(&nvmultiurisrcbin->bin_lock);
            gst_nvmultiurisrcbincreator_src_config_free(sourceConfig);
            return;
        }

        /** Add the source */
        nvmultiurisrcbin->config->uri = (gchar *)stream_info->value_camera_url.c_str();
        nvmultiurisrcbin->config->sensorId = (gchar *)stream_info->value_camera_id.c_str();
        nvmultiurisrcbin->config->source_id = 0;

        gboolean ret = gst_nvmultiurisrcbincreator_add_source(
            nvmultiurisrcbin->nvmultiurisrcbinCreator, nvmultiurisrcbin->config);
        if (ret == FALSE) {
            GST_WARNING_OBJECT(nvmultiurisrcbin, "Failed to add sensor id=[%s] uri=[%s]\n",
                               nvmultiurisrcbin->config->sensorId, nvmultiurisrcbin->config->uri);
            stream_info->status = STREAM_ADD_FAIL;
        } else {
            GST_DEBUG_OBJECT(nvmultiurisrcbin, "Successfully added sensor id=[%s] uri=[%s]\n",
                             nvmultiurisrcbin->config->sensorId, nvmultiurisrcbin->config->uri);
            stream_info->status = STREAM_ADD_SUCCESS;
        }
        gst_nvmultiurisrcbincreator_sync_children_states(nvmultiurisrcbin->nvmultiurisrcbinCreator);
        gst_element_call_async(GST_ELEMENT(nvmultiurisrcbin),
                               (GstElementCallAsyncFunc)gst_bin_sync_children_states, NULL, NULL);
        /** clean the config place-holders that we change for each source */
        nvmultiurisrcbin->config->uri = NULL;
        nvmultiurisrcbin->config->sensorId = NULL;
    } else if (g_strrstr(stream_info->value_change.c_str(), "remove")) {
        /** Remove the source */
        /** First, find the GstDsNvUriSrcConfig object from nvmultiurisrcbinCreator
         * for the provided sensorId and uri */
        GstDsNvUriSrcConfig const *sourceConfig = gst_nvmultiurisrcbincreator_get_source_config(
            nvmultiurisrcbin->nvmultiurisrcbinCreator, stream_info->value_camera_url.c_str(),
            stream_info->value_camera_id.c_str());
        if (sourceConfig) {
            gboolean ret = gst_nvmultiurisrcbincreator_remove_source(
                nvmultiurisrcbin->nvmultiurisrcbinCreator, sourceConfig->source_id);
            // Note: after call to gst_nvmultiurisrcbincreator_remove_source, sourceConfig will be
            // invalid
            sourceConfig = NULL;
            if (ret == FALSE) {
                GST_WARNING_OBJECT(nvmultiurisrcbin, "Failed to remove sensor\n");
                stream_info->status = STREAM_REMOVE_FAIL;
            } else {
                GST_DEBUG_OBJECT(nvmultiurisrcbin, "Successfully removed sensor\n");
                stream_info->status = STREAM_REMOVE_SUCCESS;
                gst_nvmultiurisrcbincreator_sync_children_states(
                    nvmultiurisrcbin->nvmultiurisrcbinCreator);
                gst_element_call_async(GST_ELEMENT(nvmultiurisrcbin),
                                       (GstElementCallAsyncFunc)gst_bin_sync_children_states, NULL,
                                       NULL);
            }
        } else {
            GST_WARNING_OBJECT(
                nvmultiurisrcbin, "No record found; Failed to remove sensor id=[%s] uri=[%s]\n",
                stream_info->value_camera_id.c_str(), stream_info->value_camera_url.c_str());
        }
    } else {
        GST_WARNING_OBJECT(nvmultiurisrcbin, "Sensor API change string not supported\n");
    }
    g_mutex_unlock(&nvmultiurisrcbin->bin_lock);
    return;
}

static void s_roi_api_impl(NvDsRoiInfo *roi_info, void *ctx)
{
    GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *)ctx;
    (void)nvmultiurisrcbin;

    guint sourceId = std::stoi(roi_info->stream_id);

    if (!find_source(nvmultiurisrcbin->nvmultiurisrcbinCreator, sourceId)) {
        roi_info->status = ROI_UPDATE_FAIL;
    } else {
        RoiDimension roi_dim[roi_info->roi_count];

        for (int i = 0; i < (int)roi_info->roi_count; i++) {
            g_strlcpy(roi_dim[i].roi_id, roi_info->vect[i].roi_id, sizeof(roi_dim[i].roi_id));
            roi_dim[i].left = roi_info->vect[i].left;
            roi_dim[i].top = roi_info->vect[i].top;
            roi_dim[i].width = roi_info->vect[i].width;
            roi_dim[i].height = roi_info->vect[i].height;
        }

        GstEvent *nvevent = gst_nvevent_new_roi_update((char *)roi_info->stream_id.c_str(),
                                                       roi_info->roi_count, roi_dim);

        if (!nvevent) {
            roi_info->roi_log = "nv-roi-update event creation failed";
            roi_info->status = ROI_UPDATE_FAIL;
        }

        if (!gst_pad_push_event((GstPad *)(nvmultiurisrcbin->bin_src_pad), nvevent)) {
            switch (roi_info->roi_flag) {
            case ROI_UPDATE:
                g_print("[WARN] nv-roi-update event not pushed downstream.. !! \n");
                roi_info->roi_log = "nv-roi-update event not pushed";
                roi_info->status = ROI_UPDATE_FAIL;
                break;
            default:
                break;
            }
        } else {
            switch (roi_info->roi_flag) {
            case ROI_UPDATE:
                roi_info->status = ROI_UPDATE_SUCCESS;
                break;
            default:
                break;
            }
        }

        gst_nvmultiurisrcbincreator_sync_children_states(nvmultiurisrcbin->nvmultiurisrcbinCreator);
        gst_element_call_async(GST_ELEMENT(nvmultiurisrcbin),
                               (GstElementCallAsyncFunc)gst_bin_sync_children_states, NULL, NULL);
    }
}

static void s_dec_api_impl(NvDsDecInfo *dec_info, void *ctx)
{
    GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *)ctx;
    (void)nvmultiurisrcbin;
    guint sourceId = std::stoi(dec_info->stream_id);

    if (!set_nvuribin_dec_prop(nvmultiurisrcbin->nvmultiurisrcbinCreator, sourceId, dec_info)) {
        switch (dec_info->dec_flag) {
        case DROP_FRAME_INTERVAL:
            g_print("[WARN] drop-frame-interval not set on decoder .. !! \n");
            dec_info->status = DROP_FRAME_INTERVAL_UPDATE_FAIL;
            break;
        case SKIP_FRAMES:
            g_print("[WARN] skip-frame not set on decoder .. !! \n");
            dec_info->status = SKIP_FRAMES_UPDATE_FAIL;
            break;
        case LOW_LATENCY_MODE:
            g_print("[WARN] low-latency-mode not set on decoder .. !! \n");
            dec_info->status = LOW_LATENCY_MODE_UPDATE_FAIL;
            break;
        default:
            break;
        }
    } else {
        switch (dec_info->dec_flag) {
        case DROP_FRAME_INTERVAL:
            dec_info->status = dec_info->status != DROP_FRAME_INTERVAL_UPDATE_FAIL
                                   ? DROP_FRAME_INTERVAL_UPDATE_SUCCESS
                                   : DROP_FRAME_INTERVAL_UPDATE_FAIL;
            break;
        case SKIP_FRAMES:
            dec_info->status = dec_info->status != SKIP_FRAMES_UPDATE_FAIL
                                   ? SKIP_FRAMES_UPDATE_SUCCESS
                                   : SKIP_FRAMES_UPDATE_FAIL;
            break;
        case LOW_LATENCY_MODE:
            dec_info->status = dec_info->status != LOW_LATENCY_MODE_UPDATE_FAIL
                                   ? LOW_LATENCY_MODE_UPDATE_SUCCESS
                                   : LOW_LATENCY_MODE_UPDATE_FAIL;
            break;
        default:
            break;
        }
    }

    gst_nvmultiurisrcbincreator_sync_children_states(nvmultiurisrcbin->nvmultiurisrcbinCreator);
    gst_element_call_async(GST_ELEMENT(nvmultiurisrcbin),
                           (GstElementCallAsyncFunc)gst_bin_sync_children_states, NULL, NULL);
}

static void s_infer_api_impl(NvDsInferInfo *infer_info, void *ctx)
{
    GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *)ctx;
    (void)nvmultiurisrcbin;
    guint sourceId = std::stoi(infer_info->stream_id);

    if (!find_source(nvmultiurisrcbin->nvmultiurisrcbinCreator, sourceId)) {
        infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
    } else {
        GstEvent *nvevent = gst_nvevent_infer_interval_update((char *)infer_info->stream_id.c_str(),
                                                              infer_info->interval);

        if (!nvevent) {
            infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
            infer_info->infer_log = "nv-infer-interval-update event creation failed";
        }

        if (!gst_pad_push_event((GstPad *)(nvmultiurisrcbin->bin_src_pad), nvevent)) {
            g_print("[WARN] nv-infer-interval-update event not pushed downstream.. !! \n");
            infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
            infer_info->infer_log = "nv-infer-interval-update event not pushed";
        } else {
            if (infer_info->status != INFER_INTERVAL_UPDATE_FAIL)
                infer_info->status = INFER_INTERVAL_UPDATE_SUCCESS;
        }

        gst_nvmultiurisrcbincreator_sync_children_states(nvmultiurisrcbin->nvmultiurisrcbinCreator);
        gst_element_call_async(GST_ELEMENT(nvmultiurisrcbin),
                               (GstElementCallAsyncFunc)gst_bin_sync_children_states, NULL, NULL);
    }
}

static void s_inferserver_api_impl(NvDsInferServerInfo *inferserver_info, void *ctx)
{
    GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *)ctx;
    (void)nvmultiurisrcbin;
    guint sourceId = std::stoi(inferserver_info->stream_id);

    if (!find_source(nvmultiurisrcbin->nvmultiurisrcbinCreator, sourceId)) {
        if (inferserver_info->inferserver_flag == INFERSERVER_INTERVAL)
            inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
    } else {
        GstEvent *nvevent = gst_nvevent_infer_interval_update(
            (char *)inferserver_info->stream_id.c_str(), inferserver_info->interval);
        if (!nvevent) {
            inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
            inferserver_info->inferserver_log =
                "nv-infer-interval-update event (inferserver) creation failed";
        }

        if (!gst_pad_push_event((GstPad *)(nvmultiurisrcbin->bin_src_pad), nvevent)) {
            g_print(
                "[WARN] nv-infer-interval-update (inferserver) event not pushed downstream.. !! "
                "\n");
            inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
            inferserver_info->inferserver_log = "nv-infer-interval-update event not pushed";
        } else {
            inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_SUCCESS;
        }
        gst_nvmultiurisrcbincreator_sync_children_states(nvmultiurisrcbin->nvmultiurisrcbinCreator);
        gst_element_call_async(GST_ELEMENT(nvmultiurisrcbin),
                               (GstElementCallAsyncFunc)gst_bin_sync_children_states, NULL, NULL);
    }
}

static void s_conv_api_impl(NvDsConvInfo *conv_info, void *ctx)
{
    GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *)ctx;
    (void)nvmultiurisrcbin;
    guint sourceId = std::stoi(conv_info->stream_id);

    if (!set_nvuribin_conv_prop(nvmultiurisrcbin->nvmultiurisrcbinCreator, sourceId, conv_info)) {
        switch (conv_info->conv_flag) {
        case SRC_CROP:
            g_print("[WARN] source-crop update failed .. !! \n");
            conv_info->conv_log = "source-crop update failed";
            conv_info->status = SRC_CROP_UPDATE_FAIL;
            break;
        case DEST_CROP:
            g_print("[WARN] source-crop update failed .. !! \n");
            conv_info->conv_log = "dest-crop update failed";
            conv_info->status = DEST_CROP_UPDATE_FAIL;
            break;
        case FLIP_METHOD:
            g_print("[WARN] flip-method update failed .. !! \n");
            conv_info->conv_log = "flip-method update failed";
            conv_info->status = FLIP_METHOD_UPDATE_FAIL;
            break;
        case INTERPOLATION_METHOD:
            g_print("[WARN] interpolation-method update failed .. !! \n");
            conv_info->conv_log = "interpolation-method update failed";
            conv_info->status = INTERPOLATION_METHOD_UPDATE_FAIL;
            break;
        default:
            break;
        }
    } else {
        switch (conv_info->conv_flag) {
        case SRC_CROP:
            conv_info->status = conv_info->status != SRC_CROP_UPDATE_FAIL ? SRC_CROP_UPDATE_SUCCESS
                                                                          : SRC_CROP_UPDATE_FAIL;
            break;
        case DEST_CROP:
            conv_info->status = conv_info->status != SRC_CROP_UPDATE_FAIL ? DEST_CROP_UPDATE_SUCCESS
                                                                          : DEST_CROP_UPDATE_FAIL;
            break;
        case FLIP_METHOD:
            conv_info->status = conv_info->status != FLIP_METHOD_UPDATE_FAIL
                                    ? FLIP_METHOD_UPDATE_SUCCESS
                                    : FLIP_METHOD_UPDATE_FAIL;
            break;
        case INTERPOLATION_METHOD:
            conv_info->status = conv_info->status != INTERPOLATION_METHOD_UPDATE_FAIL
                                    ? INTERPOLATION_METHOD_UPDATE_SUCCESS
                                    : INTERPOLATION_METHOD_UPDATE_FAIL;
            break;
        default:
            break;
        }
    }

    gst_nvmultiurisrcbincreator_sync_children_states(nvmultiurisrcbin->nvmultiurisrcbinCreator);
    gst_element_call_async(GST_ELEMENT(nvmultiurisrcbin),
                           (GstElementCallAsyncFunc)gst_bin_sync_children_states, NULL, NULL);
}

static void s_mux_api_impl(NvDsMuxInfo *mux_info, void *ctx)
{
    GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *)ctx;
    (void)nvmultiurisrcbin;

    if (!set_nvuribin_mux_prop(nvmultiurisrcbin->nvmultiurisrcbinCreator, mux_info)) {
        switch (mux_info->mux_flag) {
        case BATCHED_PUSH_TIMEOUT:
            g_print("[WARN] batched-push-timeout update failed .. !! \n");
            mux_info->mux_log = "batched-push-timeout value not updated";
            mux_info->status = BATCHED_PUSH_TIMEOUT_UPDATE_FAIL;
            break;
        case MAX_LATENCY:
            g_print("[WARN] max-latency update failed .. !! \n");
            mux_info->mux_log = "max-latency value not updated";
            mux_info->status = MAX_LATENCY_UPDATE_FAIL;
            break;
        default:
            break;
        }
    } else {
        switch (mux_info->mux_flag) {
        case BATCHED_PUSH_TIMEOUT:
            mux_info->status = mux_info->status != BATCHED_PUSH_TIMEOUT_UPDATE_FAIL
                                   ? BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS
                                   : BATCHED_PUSH_TIMEOUT_UPDATE_FAIL;
            break;
        case MAX_LATENCY:
            mux_info->status = mux_info->status != MAX_LATENCY_UPDATE_FAIL
                                   ? MAX_LATENCY_UPDATE_SUCCESS
                                   : MAX_LATENCY_UPDATE_FAIL;
            break;
        default:
            break;
        }
    }

    gst_nvmultiurisrcbincreator_sync_children_states(nvmultiurisrcbin->nvmultiurisrcbinCreator);
    gst_element_call_async(GST_ELEMENT(nvmultiurisrcbin),
                           (GstElementCallAsyncFunc)gst_bin_sync_children_states, NULL, NULL);
}

static void s_enc_api_impl(NvDsEncInfo *enc_info, void *ctx)
{
    GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *)ctx;
    (void)nvmultiurisrcbin;
    guint sourceId = std::stoi(enc_info->stream_id);
    GstEvent *nvevent = NULL;

    if (!find_source(nvmultiurisrcbin->nvmultiurisrcbinCreator, sourceId)) {
        switch (enc_info->enc_flag) {
        case BITRATE:
            enc_info->enc_log = "Not able to find source id";
            enc_info->status = BITRATE_UPDATE_FAIL;
            break;
        case FORCE_IDR:
            enc_info->enc_log = "Not able to find source id";
            enc_info->status = FORCE_IDR_UPDATE_FAIL;
            break;
        case FORCE_INTRA:
            enc_info->enc_log = "Not able to find source id";
            enc_info->status = FORCE_INTRA_UPDATE_FAIL;
            break;
        case IFRAME_INTERVAL:
            enc_info->enc_log = "Not able to find source id";
            enc_info->status = IFRAME_INTERVAL_UPDATE_FAIL;
            break;
        default:
            break;
        }
    } else {
        switch (enc_info->enc_flag) {
        case BITRATE:
            nvevent = gst_nvevent_enc_bitrate_update((char *)enc_info->stream_id.c_str(),
                                                     enc_info->bitrate);
            if (!gst_pad_push_event((GstPad *)(nvmultiurisrcbin->bin_src_pad), nvevent)) {
                g_print(
                    "[WARN] nv-enc-bitrate-update event not pushed downstream.bitrate update "
                    "failed on encoder.. !! \n");
                enc_info->enc_log = !nvevent ? "nv-enc-bitrate-update event creation failed"
                                             : "nv-enc-bitrate-update event not pushed";
                enc_info->status = BITRATE_UPDATE_FAIL;
            } else {
                enc_info->status = BITRATE_UPDATE_SUCCESS;
            }
            break;
        case FORCE_IDR:
            nvevent =
                gst_nvevent_enc_force_idr((char *)enc_info->stream_id.c_str(), enc_info->force_idr);
            if (!gst_pad_push_event((GstPad *)(nvmultiurisrcbin->bin_src_pad), nvevent)) {
                g_print(
                    "[WARN] nv-enc-force-idr event not pushed downstream.force IDR frame failed on "
                    "encoder .. !! \n");
                enc_info->enc_log = !nvevent ? "nv-enc-force-idr event creation failed"
                                             : "nv-enc-force-idr event not pushed";
                enc_info->status = FORCE_IDR_UPDATE_FAIL;
            } else {
                enc_info->status = FORCE_IDR_UPDATE_SUCCESS;
            }
            break;
        case FORCE_INTRA:
            nvevent = gst_nvevent_enc_force_intra((char *)enc_info->stream_id.c_str(),
                                                  enc_info->force_intra);
            if (!gst_pad_push_event((GstPad *)(nvmultiurisrcbin->bin_src_pad), nvevent)) {
                g_print(
                    "[WARN] nv-enc-force-intra event not pushed downstream.force intra frame "
                    "failed on encoder .. !! \n");
                enc_info->enc_log = !nvevent ? "nv-enc-force-intra event creation failed"
                                             : "nv-enc-force-intra event not pushed";
                enc_info->status = FORCE_INTRA_UPDATE_FAIL;
            } else {
                enc_info->status = FORCE_INTRA_UPDATE_SUCCESS;
            }
            break;
        case IFRAME_INTERVAL:
            nvevent = gst_nvevent_enc_iframeinterval_update((char *)enc_info->stream_id.c_str(),
                                                            enc_info->iframeinterval);
            if (!gst_pad_push_event((GstPad *)(nvmultiurisrcbin->bin_src_pad), nvevent)) {
                g_print(
                    "[WARN] nv-enc-iframeinterval-update event not pushed downstream.iframe "
                    "interval update failed on encoder .. !! \n");
                enc_info->enc_log = !nvevent ? "nv-enc-iframeinterval-update event creation failed"
                                             : "nv-enc-iframeinterval-update event not pushed";
                enc_info->status = IFRAME_INTERVAL_UPDATE_FAIL;
            } else {
                enc_info->status = IFRAME_INTERVAL_UPDATE_SUCCESS;
            }
            break;
        default:
            break;
        }
    }

    gst_nvmultiurisrcbincreator_sync_children_states(nvmultiurisrcbin->nvmultiurisrcbinCreator);
    gst_element_call_async(GST_ELEMENT(nvmultiurisrcbin),
                           (GstElementCallAsyncFunc)gst_bin_sync_children_states, NULL, NULL);
}

static void s_osd_api_impl(NvDsOsdInfo *osd_info, void *ctx)
{
    GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *)ctx;
    (void)nvmultiurisrcbin;

    guint sourceId = std::stoi(osd_info->stream_id);

    if (!find_source(nvmultiurisrcbin->nvmultiurisrcbinCreator, sourceId)) {
        if (osd_info->osd_flag == PROCESS_MODE)
            osd_info->status = PROCESS_MODE_UPDATE_FAIL;
    } else {
        GstEvent *nvevent = gst_nvevent_osd_process_mode_update((char *)osd_info->stream_id.c_str(),
                                                                osd_info->process_mode);
        if (!nvevent) {
            osd_info->status = PROCESS_MODE_UPDATE_FAIL;
            osd_info->osd_log = "nv-osd-process-mode-update event creation failed";
        }

        if (!gst_pad_push_event((GstPad *)(nvmultiurisrcbin->bin_src_pad), nvevent)) {
            g_print("[WARN] nv-osd-process-mode-update event not pushed downstream.. !! \n");
            osd_info->status = PROCESS_MODE_UPDATE_FAIL;
            osd_info->osd_log = "nv-osd-process-mode-update event not pushed";
        } else {
            osd_info->status = PROCESS_MODE_UPDATE_SUCCESS;
        }
    }
}

static void s_appinstance_api_impl(NvDsAppInstanceInfo *appinstance_info, void *ctx)
{
    GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *)ctx;
    (void)nvmultiurisrcbin;

    if (!s_force_eos_handle(nvmultiurisrcbin->nvmultiurisrcbinCreator, appinstance_info)) {
        if (appinstance_info->appinstance_flag == QUIT_APP) {
            appinstance_info->app_log = "Unable to handle force-pipeline-eos nvmultiurisrcbin";
            appinstance_info->status = QUIT_FAIL;
        }
    } else {
        if (appinstance_info->appinstance_flag == QUIT_APP) {
            appinstance_info->status = QUIT_SUCCESS;
        }
    }
}

static void rest_api_server_start(GstDsNvMultiUriBin *nvmultiurisrcbin)
{
    NvDsServerCallbacks server_cb = {};

    server_cb.stream_cb = [nvmultiurisrcbin](NvDsStreamInfo *stream_info, void *ctx) {
        s_stream_api_impl(stream_info, (void *)nvmultiurisrcbin);
    };
    server_cb.roi_cb = [nvmultiurisrcbin](NvDsRoiInfo *roi_info, void *ctx) {
        s_roi_api_impl(roi_info, (void *)nvmultiurisrcbin);
    };
    server_cb.dec_cb = [nvmultiurisrcbin](NvDsDecInfo *dec_info, void *ctx) {
        s_dec_api_impl(dec_info, (void *)nvmultiurisrcbin);
    };
    server_cb.infer_cb = [nvmultiurisrcbin](NvDsInferInfo *infer_info, void *ctx) {
        s_infer_api_impl(infer_info, (void *)nvmultiurisrcbin);
    };
    server_cb.inferserver_cb = [nvmultiurisrcbin](NvDsInferServerInfo *inferserver_info,
                                                  void *ctx) {
        s_inferserver_api_impl(inferserver_info, (void *)nvmultiurisrcbin);
    };
    server_cb.conv_cb = [nvmultiurisrcbin](NvDsConvInfo *conv_info, void *ctx) {
        s_conv_api_impl(conv_info, (void *)nvmultiurisrcbin);
    };
    server_cb.enc_cb = [nvmultiurisrcbin](NvDsEncInfo *enc_info, void *ctx) {
        s_enc_api_impl(enc_info, (void *)nvmultiurisrcbin);
    };
    server_cb.mux_cb = [nvmultiurisrcbin](NvDsMuxInfo *mux_info, void *ctx) {
        s_mux_api_impl(mux_info, (void *)nvmultiurisrcbin);
    };

    server_cb.osd_cb = [nvmultiurisrcbin](NvDsOsdInfo *osd_info, void *ctx) {
        s_osd_api_impl(osd_info, (void *)nvmultiurisrcbin);
    };

    server_cb.appinstance_cb = [nvmultiurisrcbin](NvDsAppInstanceInfo *appinstance_info,
                                                  void *ctx) {
        s_appinstance_api_impl(appinstance_info, (void *)nvmultiurisrcbin);
    };

    NvDsServerConfig server_conf = {};
    server_conf.ip = std::string(nvmultiurisrcbin->httpIp);
    server_conf.port = std::string(nvmultiurisrcbin->httpPort);
    nvmultiurisrcbin->restServer = (void *)nvds_rest_server_start(&server_conf, &server_cb);
}

#ifdef ENABLE_GST_NVMULTIURISRCBIN_UNIT_TESTS

#ifndef PACKAGE
#define PACKAGE "nvdsbins"
#endif

#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA Multiurisrcbin"
#define BINARY_PACKAGE "NVIDIA DeepStream Bins"
#define URL "http://nvidia.com/"

extern "C" gboolean plugin_init_2(GstPlugin *plugin)
{
    if (!gst_element_register(plugin, "nvmultiurisrcbin", GST_RANK_PRIMARY,
                              GST_TYPE_DS_NVMULTIURISRC_BIN))
        return FALSE;

    return TRUE;
}

extern "C" gboolean gGstNvMultiUriSrcBinStaticInit();
gboolean gGstNvMultiUriSrcBinStaticInit()
{
    return gst_plugin_register_static(GST_VERSION_MAJOR, GST_VERSION_MINOR,
                                      "nvdsgst_deepstream_bins2", DESCRIPTION, plugin_init_2, "6.3",
                                      LICENSE, BINARY_PACKAGE, PACKAGE, URL);
}
#endif
