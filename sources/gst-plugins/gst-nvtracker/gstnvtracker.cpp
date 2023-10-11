/**
 * Copyright (c) 2016-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 * version: 0.2
 */

#include "gstnvtracker.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <gst/base/base.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "gst-nvcommon.h"
#include "gst-nvevent.h"
#include "gst-nvquery.h"
#include "logging.h"
#include "nvds_dewarper_meta.h"
#include "nvtracker_proc.h"

using namespace std;

/* Filter signals and args */
enum {
    PROP_0,
    PROP_TRACKER_WIDTH,
    PROP_TRACKER_HEIGHT,
    PROP_GPU_DEVICE_ID,
    PROP_CONFIG_PATH,
    PROP_LL_CONFIG_PATH,
    PROP_LL_LIB_PATH,
    PROP_SELECTIVE_TRACKING,
    PROP_COMPUTE_HW,
    PROP_DISPLAY_TRACK_ID,
    PROP_TRACK_ID_RESET_MODE,
    PROP_INPUT_TENSOR_META,
    PROP_TENSOR_META_GIE_ID,
    PROP_USER_META_POOL_SIZE
};

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate nvtracker_sink_factory = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM,
                                                      "{ "
                                                      "I420, NV12, RGBA }")));

static GstStaticPadTemplate nvtracker_src_factory = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM,
                                                      "{ "
                                                      "I420, NV12, RGBA }")));

/* Default gstreamer property values */
#define DEFAULT_LL_CONFIG_FILE NULL
#define DEFAULT_LL_LIB_FILE NULL
#define DEFAULT_INPUT_TENSOR_META FALSE
#define DEFAULT_DISPLAY_TRACKING_ID TRUE
#define DEFAULT_TRACKER_WIDTH 960
#define DEFAULT_TRACKER_HEIGHT 544
#define DEFAULT_TRACKER_GIE_ID -1
#define EXPECTED_FORMAT "RGBA"
#define DEFAULT_GPU_ID 0
#define DEFAULT_CONV_BUF_POOL_SIZE 4
#define DEFAULT_USER_META_POOL_SIZE 16

static GQuark _dsmeta_quark;

#define gst_nv_tracker_parent_class parent_class
G_DEFINE_TYPE(GstNvTracker, gst_nv_tracker, GST_TYPE_BASE_TRANSFORM);

GST_DEBUG_CATEGORY_STATIC(gst_nv_tracker_debug);
#define GST_CAT_DEFAULT gst_nv_tracker_debug

static void gst_nv_tracker_finalize(GObject *object);
static void gst_nv_tracker_set_property(GObject *object,
                                        guint prop_id,
                                        const GValue *value,
                                        GParamSpec *pspec);
static void gst_nv_tracker_get_property(GObject *object,
                                        guint prop_id,
                                        GValue *value,
                                        GParamSpec *pspec);
static GstFlowReturn gst_nv_tracker_submit_input_buffer(GstBaseTransform *trans,
                                                        gboolean is_discont,
                                                        GstBuffer *buf);
static GstFlowReturn gst_nv_tracker_generate_output(GstBaseTransform *trans, GstBuffer **outbuf);
static gboolean gst_nv_tracker_start(GstBaseTransform *btrans);
static gboolean gst_nv_tracker_stop(GstBaseTransform *btrans);

static gpointer gst_nv_nvtracker_output_loop(gpointer user_data);

static gboolean gst_nv_tracker_set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
    return TRUE;
}

static gboolean gst_nv_tracker_sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
    gboolean ret = TRUE;
    bool result = true;
    gboolean ignore_serialized_event = FALSE;

    GstBaseTransform *trans = NULL;
    GstBaseTransformClass *bclass = NULL;
    GstNvTracker *nvtracker = NULL;
    guint source_id = 0;

    trans = GST_BASE_TRANSFORM(parent);
    bclass = GST_BASE_TRANSFORM_GET_CLASS(trans);
    nvtracker = GST_NVTRACKER(trans);
    /* The TAG event is sent many times leading to drop in performance because of
     * buffer/event serialization. We can ignore such events which won't cause
     * issues if we don't serialize the events. */
    switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_TAG:
        ignore_serialized_event = TRUE;
        break;
    default:
        break;
    }

    /* Serialize events. Wait for pending buffers to be processed and pushed
     * downstream.
     */
    if (GST_EVENT_IS_SERIALIZED(event) && !ignore_serialized_event && nvtracker->output_loop) {
        g_mutex_lock(&nvtracker->eventLock);
        result = nvtracker->trackerIface->flushReqs();
        g_cond_wait(&nvtracker->eventCondition, &nvtracker->eventLock);
        g_mutex_unlock(&nvtracker->eventLock);
    }

    switch (static_cast<GstNvEventType>(event->type)) {
    case GST_NVEVENT_PAD_ADDED:
        gst_nvevent_parse_pad_added(event, &source_id);
        GST_DEBUG_OBJECT(nvtracker, "Pad added %d\n", source_id);
        result = nvtracker->trackerIface->addSource(source_id);
        break;
    case GST_NVEVENT_PAD_DELETED:
        gst_nvevent_parse_pad_deleted(event, &source_id);
        /** Stream is really deleted, remove everything */
        GST_DEBUG_OBJECT(nvtracker, "Pad deleted %d\n", source_id);
        result = nvtracker->trackerIface->removeSource(source_id);
        break;
    case GST_NVEVENT_STREAM_RESET:
        gst_nvevent_parse_stream_reset(event, &source_id);
        if (nvtracker->trackerConfig.trackingIdResetMode &
            TrackingIdResetMode_NewIdAfterStreamReset) {
            /** Remove existing trackers, but don't change objectIdMapping */
            result = nvtracker->trackerIface->removeSource(source_id, false);
        }
        break;
    case GST_NVEVENT_STREAM_EOS:
        gst_nvevent_parse_stream_eos(event, &source_id);
        if (nvtracker->trackerConfig.trackingIdResetMode & TrackingIdResetMode_FromZeroAfterEOS) {
            /** Remove existing trackers, remove objectIdMapping so in the new loop a new
             * objectIdOffset will be set */
            result = nvtracker->trackerIface->removeSource(source_id);
        }
        break;
    case GST_NVEVENT_STREAM_SEGMENT:
    default:
        break;
    }

    if (!result) {
        GST_ERROR("gstnvtracker: Failed to handle event\n");
        return FALSE;
    }

    if (bclass->sink_event) {
        ret = bclass->sink_event(trans, event);
    }

    return ret;
}

static gboolean gst_nv_tracker_start(GstBaseTransform *btrans)
{
    GstNvTracker *nvtracker = GST_NVTRACKER(btrans);

    GstQuery *nsquery = gst_nvquery_numStreams_size_new();
    guint numStreams = 1;
    if (gst_pad_peer_query(GST_BASE_TRANSFORM_SINK_PAD(btrans), nsquery)) {
        gst_nvquery_numStreams_size_parse(nsquery, &numStreams);
        GST_DEBUG_OBJECT(nvtracker, "gstnvtracker: numStreams set as %d...\n", 0);
    } else {
        GST_DEBUG_OBJECT(nvtracker, "gstnvtracker: numStreams not set. so setting default to 1\n");
    }
    gst_query_unref(nsquery);

    GstQuery *bsquery = gst_nvquery_batch_size_new();
    guint batchSize = 1;
    if (gst_pad_peer_query(GST_BASE_TRANSFORM_SINK_PAD(btrans), bsquery)) {
        gst_nvquery_batch_size_parse(bsquery, &batchSize);
        nvtracker->trackerConfig.batchSize = batchSize;
        GST_DEBUG_OBJECT(nvtracker, "gstnvtracker: batchSize set as %d...\n",
                         nvtracker->trackerConfig.batchSize);
    } else {
        GST_DEBUG_OBJECT(nvtracker, "gstnvtracker: batchSize not set. so setting default to 1\n");
        nvtracker->trackerConfig.batchSize = 1;
    }
    gst_query_unref(bsquery);

    if (nvtracker->trackerConfig.inputTensorMeta == true) {
        GstQuery *poolsizequery =
            gst_nvquery_preprocess_poolsize_new(nvtracker->trackerConfig.tensorMetaGieId);
        guint poolSize = 4;
        if (gst_pad_peer_query(GST_BASE_TRANSFORM_SINK_PAD(btrans), poolsizequery)) {
            gst_nvquery_preprocess_poolsize_parse(poolsizequery, &poolSize);
            nvtracker->trackerConfig.maxConvBufPoolSize = poolSize;
            GST_DEBUG_OBJECT(nvtracker, "gstnvtracker: poolSize query set as %d...\n", poolSize);
        } else {
            GST_DEBUG_OBJECT(nvtracker,
                             "gstnvtracker: poolSize query failed. so setting default to 4\n");
            nvtracker->trackerConfig.maxConvBufPoolSize = 4;
        }
        gst_query_unref(poolsizequery);
    }

    nvtracker->running = FALSE;

    cudaError_t cudaReturn = cudaSetDevice(nvtracker->trackerConfig.gpuId);
    if (cudaReturn != cudaSuccess) {
        GST_ERROR("gstnvtracker: Failed to set gpu-id with error: %s\n",
                  cudaGetErrorName(cudaReturn));
        return FALSE;
    }
    nvtracker->trackerIface = new (std::nothrow) NvTrackerProc();
    if (NULL == nvtracker->trackerIface) {
        GST_ERROR("gstnvtracker: Failed to allocate trackerIface\n");
        return FALSE;
    }

    bool initResult = nvtracker->trackerIface->init(nvtracker->trackerConfig);
    if (!initResult) {
        GST_ERROR("gstnvtracker: Failed to initialize trackerIface\n");
        delete nvtracker->trackerIface;
        nvtracker->trackerIface = NULL;
        return FALSE;
    }

    gst_pad_set_event_function(GST_BASE_TRANSFORM_SINK_PAD(nvtracker),
                               GST_DEBUG_FUNCPTR(gst_nv_tracker_sink_event));

    nvtracker->running = TRUE;

    nvtracker->output_loop = g_thread_new("gst_nv_nvtracker_output_loop",
                                          gst_nv_nvtracker_output_loop, (gpointer)nvtracker);

    return TRUE;
}

static gboolean gst_nv_tracker_stop(GstBaseTransform *btrans)
{
    GstNvTracker *nvtracker = GST_NVTRACKER(btrans);

    /** De-init the low-level threads and plugin */
    nvtracker->trackerIface->deInit();

    /** Terminate the output loop
     * Note: This needs to be done AFTER plugin deInit() process
     * so that all the pending buffers can be returned properly
     */
    nvtracker->running = FALSE;

    if (nvtracker->output_loop) {
        g_thread_join(nvtracker->output_loop);
    }
    nvtracker->output_loop = NULL;

    delete nvtracker->trackerIface;
    nvtracker->trackerIface = NULL;

    return TRUE;
}

static void gst_nv_tracker_finalize(GObject *object)
{
    GstNvTracker *nvtracker = GST_NVTRACKER(object);

    if (nvtracker->trackerConfig.trackerLibFile != NULL) {
        g_free(nvtracker->trackerConfig.trackerLibFile);
        nvtracker->trackerConfig.trackerLibFile = NULL;
    }

    if (nvtracker->trackerConfig.trackerConfigFile != NULL) {
        g_free(nvtracker->trackerConfig.trackerConfigFile);
        nvtracker->trackerConfig.trackerConfigFile = NULL;
    }

    if (nvtracker->trackerConfig.gstName != NULL) {
        g_free(nvtracker->trackerConfig.gstName);
        nvtracker->trackerConfig.gstName = NULL;
    }

    g_cond_clear(&nvtracker->eventCondition);
    g_mutex_clear(&nvtracker->eventLock);
    G_OBJECT_CLASS(parent_class)->finalize(object);
}

/* initialize the nvtracker's class */
static void gst_nv_tracker_class_init(GstNvTrackerClass *klass)
{
    GObjectClass *gobject_class;
    GstElementClass *gstelement_class;
    GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);

    gobject_class = (GObjectClass *)klass;
    gstelement_class = (GstElementClass *)klass;

    base_transform_class->submit_input_buffer =
        GST_DEBUG_FUNCPTR(gst_nv_tracker_submit_input_buffer);
    base_transform_class->generate_output = GST_DEBUG_FUNCPTR(gst_nv_tracker_generate_output);

    /**  base_transform_class->transform_ip = GST_DEBUG_FUNCPTR(gst_nv_tracker_transform_ip); */
    base_transform_class->start = GST_DEBUG_FUNCPTR(gst_nv_tracker_start);
    base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_nv_tracker_stop);
    base_transform_class->set_caps = GST_DEBUG_FUNCPTR(gst_nv_tracker_set_caps);

    gobject_class->set_property = gst_nv_tracker_set_property;
    gobject_class->get_property = gst_nv_tracker_get_property;
    gobject_class->finalize = gst_nv_tracker_finalize;

    base_transform_class->passthrough_on_same_caps = TRUE;

    g_object_class_install_property(
        gobject_class, PROP_TRACKER_WIDTH,
        g_param_spec_uint(
            "tracker-width", "Tracker Width",
            "Frame width at which the tracker should operate, in pixels", 0, G_MAXUINT,
            DEFAULT_TRACKER_WIDTH,
            GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, PROP_TRACKER_HEIGHT,
        g_param_spec_uint(
            "tracker-height", "Tracker Height",
            "Frame height at which the tracker should operate, in pixels", 0, G_MAXUINT,
            DEFAULT_TRACKER_HEIGHT,
            GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, PROP_GPU_DEVICE_ID,
        g_param_spec_uint(
            "gpu-id", "Set GPU Device ID", "Set GPU Device ID", 0, G_MAXUINT, DEFAULT_GPU_ID,
            GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, PROP_LL_CONFIG_PATH,
        g_param_spec_string("ll-config-file", "Low-level library config file",
                            "Low-level library config file path", DEFAULT_LL_CONFIG_FILE,
                            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_LL_LIB_PATH,
        g_param_spec_string("ll-lib-file", "Low-level library file path",
                            "Low-level library file path", DEFAULT_LL_LIB_FILE,
                            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_SELECTIVE_TRACKING,
        g_param_spec_uint(
            "tracking-surface-type", "Set Tracking Surface Type",
            "Set Tracking Surface Type, default is ALL,\
        (1) => SPOT Surface, (2) => AISLE Surface",
            0, G_MAXUINT, NVDS_META_SURFACE_NONE,
            GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    PROP_COMPUTE_HW_INSTALL(gobject_class);

    g_object_class_install_property(
        gobject_class, PROP_DISPLAY_TRACK_ID,
        g_param_spec_boolean("display-tracking-id", "Display tracking id in object text",
                             "Display tracking id in object text", DEFAULT_DISPLAY_TRACKING_ID,
                             G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_TRACK_ID_RESET_MODE,
        g_param_spec_uint(
            "tracking-id-reset-mode", "Tracking ID reset mode",
            "Tracking ID reset mode when stream reset or EOS happens", 0,
            TrackingIdResetMode_MaxValue, TrackingIdResetMode_Default,
            GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, PROP_INPUT_TENSOR_META,
        g_param_spec_boolean("input-tensor-meta",
                             "Use preprocess tensormeta if available for tensor-meta-gie-id",
                             "Use preprocess tensormeta if available for tensor-meta-gie-id",
                             DEFAULT_INPUT_TENSOR_META, G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_TENSOR_META_GIE_ID,
        g_param_spec_uint(
            "tensor-meta-gie-id",
            "Tensor Meta GIE ID to be used, property valid only if input-tensor-meta is TRUE",
            "Tensor Meta GIE ID to be used, property valid only if input-tensor-meta is TRUE", 0,
            -1, 0,
            GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, PROP_USER_META_POOL_SIZE,
        g_param_spec_uint(
            "user-meta-pool-size", "User Meta Pool Size", "Tracker user meta buffer pool size", 1,
            G_MAXUINT, DEFAULT_USER_META_POOL_SIZE,
            GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    gst_element_class_set_details_simple(
        gstelement_class, "NvTracker plugin", "NvTracker functionality",
        "Gstreamer object tracking element",
        "NVIDIA Corporation. Post on Deepstream SDK forum for any queries "
        "@ https://devtalk.nvidia.com/default/board/209/");

    gst_element_class_add_pad_template(gstelement_class,
                                       gst_static_pad_template_get(&nvtracker_src_factory));
    gst_element_class_add_pad_template(gstelement_class,
                                       gst_static_pad_template_get(&nvtracker_sink_factory));

    _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);
}

static void gst_nv_tracker_set_property(GObject *object,
                                        guint prop_id,
                                        const GValue *value,
                                        GParamSpec *pspec)
{
    GstNvTracker *nvtracker = GST_NVTRACKER(object);
    nvtracker = nvtracker;

    switch (prop_id) {
    case PROP_TRACKER_WIDTH:
        nvtracker->trackerConfig.trackerWidth = g_value_get_uint(value);
        break;
    case PROP_TRACKER_HEIGHT:
        nvtracker->trackerConfig.trackerHeight = g_value_get_uint(value);
        break;
    case PROP_GPU_DEVICE_ID:
        nvtracker->trackerConfig.gpuId = g_value_get_uint(value);
        break;
    case PROP_COMPUTE_HW:
        nvtracker->trackerConfig.compute_hw = g_value_get_enum(value);
        break;
    case PROP_LL_CONFIG_PATH:
        if (nvtracker->trackerConfig.trackerConfigFile) {
            g_free(nvtracker->trackerConfig.trackerConfigFile);
        }
        nvtracker->trackerConfig.trackerConfigFile = (char *)g_value_dup_string(value);
        break;
    case PROP_LL_LIB_PATH:
        if (nvtracker->trackerConfig.trackerLibFile) {
            g_free(nvtracker->trackerConfig.trackerLibFile);
        }
        nvtracker->trackerConfig.trackerLibFile = (char *)g_value_dup_string(value);
        break;
    case PROP_SELECTIVE_TRACKING:
        nvtracker->trackerConfig.trackingSurfType = g_value_get_uint(value);
        if (nvtracker->trackerConfig.trackingSurfType == 0) {
            nvtracker->trackerConfig.trackingSurfTypeFromConfig = false;
        } else {
            nvtracker->trackerConfig.trackingSurfTypeFromConfig = true;
        }
        break;
    case PROP_DISPLAY_TRACK_ID:
        nvtracker->trackerConfig.displayTrackingId = g_value_get_boolean(value);
        break;
    case PROP_TRACK_ID_RESET_MODE:
        nvtracker->trackerConfig.trackingIdResetMode = (TrackingIdResetMode)g_value_get_uint(value);
        break;
    case PROP_INPUT_TENSOR_META:
        nvtracker->trackerConfig.inputTensorMeta = g_value_get_boolean(value);
        break;
    case PROP_TENSOR_META_GIE_ID:
        nvtracker->trackerConfig.tensorMetaGieId = g_value_get_uint(value);
        break;
    case PROP_USER_META_POOL_SIZE:
        nvtracker->trackerConfig.maxMiscDataPoolSize = g_value_get_uint(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_nv_tracker_get_property(GObject *object,
                                        guint prop_id,
                                        GValue *value,
                                        GParamSpec *pspec)
{
    GstNvTracker *nvtracker = GST_NVTRACKER(object);
    nvtracker = nvtracker;

    switch (prop_id) {
    case PROP_TRACKER_WIDTH:
        g_value_set_uint(value, nvtracker->trackerConfig.trackerWidth);
        break;
    case PROP_TRACKER_HEIGHT:
        g_value_set_uint(value, nvtracker->trackerConfig.trackerHeight);
        break;
    case PROP_GPU_DEVICE_ID:
        g_value_set_uint(value, nvtracker->trackerConfig.gpuId);
        break;
    case PROP_COMPUTE_HW:
        g_value_set_enum(value, nvtracker->trackerConfig.compute_hw);
        break;
    case PROP_LL_CONFIG_PATH:
        g_value_set_string(value, nvtracker->trackerConfig.trackerConfigFile);
        break;
    case PROP_LL_LIB_PATH:
        g_value_set_string(value, nvtracker->trackerConfig.trackerLibFile);
        break;
    case PROP_SELECTIVE_TRACKING:
        g_value_set_uint(value, nvtracker->trackerConfig.trackingSurfType);
        break;
    case PROP_DISPLAY_TRACK_ID:
        g_value_set_boolean(value, nvtracker->trackerConfig.displayTrackingId);
        break;
    case PROP_TRACK_ID_RESET_MODE:
        g_value_set_uint(value, nvtracker->trackerConfig.trackingIdResetMode);
        break;
    case PROP_INPUT_TENSOR_META:
        g_value_set_boolean(value, nvtracker->trackerConfig.inputTensorMeta);
        break;
    case PROP_TENSOR_META_GIE_ID:
        g_value_set_uint(value, nvtracker->trackerConfig.tensorMetaGieId);
        break;
    case PROP_USER_META_POOL_SIZE:
        g_value_set_uint(value, nvtracker->trackerConfig.maxMiscDataPoolSize);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static GstFlowReturn gst_nv_tracker_submit_input_buffer(GstBaseTransform *trans,
                                                        gboolean is_discont,
                                                        GstBuffer *inbuf)
{
    GstNvTracker *nvtracker = GST_NVTRACKER(trans);

    /** NOTE: Initializing state to nullptr is essential. */
    NvDsBatchMeta *batch_meta = nullptr;
    GstMapInfo inmap;

    batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);

    if (batch_meta->num_frames_in_batch == 0) {
        g_mutex_lock(&nvtracker->eventLock);
        bool result = nvtracker->trackerIface->flushReqs();
        g_cond_wait(&nvtracker->eventCondition, &nvtracker->eventLock);
        g_mutex_unlock(&nvtracker->eventLock);
        if (!result) {
            return GST_FLOW_ERROR;
        }
        return gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD(trans), inbuf);
    }

    memset(&inmap, 0, sizeof(inmap));
    if (!gst_buffer_map(inbuf, &inmap, GST_MAP_READ)) {
        return GST_FLOW_ERROR;
    }

    nvds_set_input_system_timestamp(inbuf, GST_ELEMENT_NAME(trans));

    NvBufSurface *inputBuffer = reinterpret_cast<NvBufSurface *>(inmap.data);
    gst_buffer_unmap(inbuf, &inmap);

    /* Compose the input params and submit for tracker processing
       Keep track of the inbuf via pPreservedData, so the output loop
       can push it down the pipeline. */
    InputParams input;
    input.pSurfaceBatch = inputBuffer;
    input.pBatchMeta = batch_meta;
    input.pPreservedData = inbuf;
    input.eventMarker = false;

    if (((inputBuffer->memType == NVBUF_MEM_DEFAULT ||
          inputBuffer->memType == NVBUF_MEM_CUDA_DEVICE) &&
         ((int)inputBuffer->gpuId != (int)nvtracker->trackerConfig.gpuId)) ||
        (((int)inputBuffer->gpuId == (int)nvtracker->trackerConfig.gpuId) &&
         (inputBuffer->memType == NVBUF_MEM_SYSTEM))) {
        GST_ELEMENT_ERROR(nvtracker, RESOURCE, FAILED,
                          ("Memory Compatibility Error:Input surface gpu-id doesnt match with "
                           "configured gpu-id for element,"
                           " please allocate input using unified memory, or use same gpu-ids OR,"
                           " if same gpu-ids are used ensure appropriate Cuda memories are used"),
                          ("surface-gpu-id=%d,%s-gpu-id=%d", inputBuffer->gpuId,
                           GST_ELEMENT_NAME(nvtracker), nvtracker->trackerConfig.gpuId));
        return GST_FLOW_ERROR;
    }

    /** Check frame number in batch doesn't exceed batch size */
    if (input.pBatchMeta->num_frames_in_batch > nvtracker->trackerConfig.batchSize) {
        GST_ELEMENT_ERROR(nvtracker, STREAM, FAILED,
                          ("Frame number in input batch exceeds maximum batch size"), (nullptr));
        return GST_FLOW_ERROR;
    }

    if (!nvtracker->trackerIface->submitInput(input)) {
        GST_ELEMENT_ERROR(nvtracker, STREAM, FAILED, ("Failed to submit input to tracker"),
                          (nullptr));
        return GST_FLOW_ERROR;
    }

    return GST_FLOW_OK;
}

static gpointer gst_nv_nvtracker_output_loop(gpointer user_data)
{
    GstNvTracker *nvtracker = (GstNvTracker *)user_data;
    while (nvtracker->running) {
        InputParams inputParams;
        CompletionStatus status = nvtracker->trackerIface->waitForCompletion(inputParams);
        if (status == CompletionStatus_OK && nvtracker->running) {
            /** Check for event marker */
            if (inputParams.eventMarker) {
                g_mutex_lock(&nvtracker->eventLock);
                g_cond_signal(&nvtracker->eventCondition);
                g_mutex_unlock(&nvtracker->eventLock);
                continue;
            }
            GstBuffer *inbuf = (GstBuffer *)inputParams.pPreservedData;

            nvds_set_output_system_timestamp(inbuf, GST_ELEMENT_NAME(nvtracker));

            /** Push the buffer to peer sink pad */
            gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD(nvtracker), inbuf);

        } else if (status == CompletionStatus_Exit) {
            nvtracker->running = false;
            return nvtracker;
        }
    }

    return nvtracker;
}

/* Mandatory override of generate_output function to match submit_input.
 * The actual output is pushed from gst_nv_nvtracker_output_loop.
 */
static GstFlowReturn gst_nv_tracker_generate_output(GstBaseTransform *trans, GstBuffer **outbuf)
{
    *outbuf = NULL;
    return GST_FLOW_OK;
}

void gst_nv_tracker_init(GstNvTracker *nvtracker)
{
    /** Will be initialized from DeepStream app config file. */
    nvtracker->trackerConfig.trackerWidth = DEFAULT_TRACKER_WIDTH;
    nvtracker->trackerConfig.trackerHeight = DEFAULT_TRACKER_HEIGHT;
    nvtracker->trackerConfig.batchSize = 1;
    nvtracker->trackerConfig.trackerLibFile = NULL;
    nvtracker->trackerConfig.trackerConfigFile = NULL;

    nvtracker->trackerConfig.displayTrackingId = true;
    nvtracker->trackerConfig.trackingIdResetMode = TrackingIdResetMode_NewIdAfterStreamReset;

    nvtracker->trackerConfig.computeTarget = NVMOTCOMP_ANY;
    nvtracker->trackerConfig.gpuId = DEFAULT_GPU_ID;
    nvtracker->trackerConfig.compute_hw = NvBufSurfTransformCompute_Default;

    nvtracker->trackerConfig.trackingSurfType = NVDS_META_SURFACE_NONE;
    nvtracker->trackerConfig.trackingSurfTypeFromConfig = true;

    nvtracker->trackerConfig.inputTensorMeta = false;
    nvtracker->trackerConfig.tensorMetaGieId = 0;

    /** Will be initialized from low level tracker library query. */
    nvtracker->trackerConfig.colorFormat = NVBUF_COLOR_FORMAT_BGR;
#ifdef __aarch64__
    nvtracker->trackerConfig.memType = NVBUF_MEM_DEFAULT;
#else
    nvtracker->trackerConfig.memType = NVBUF_MEM_CUDA_DEVICE;
#endif

    nvtracker->trackerConfig.pastFrame = true;
    nvtracker->trackerConfig.numTransforms = 0;
    nvtracker->trackerConfig.maxTargetsPerStream = 150;
    nvtracker->trackerConfig.maxShadowTrackingAge = 50;
    nvtracker->trackerConfig.outputReidTensor = false;
    nvtracker->trackerConfig.reidFeatureSize = 256;

    /** Store buffer pool size since low level tracker needs this info. */
    nvtracker->trackerConfig.maxConvBufPoolSize = DEFAULT_CONV_BUF_POOL_SIZE;
    nvtracker->trackerConfig.maxMiscDataPoolSize = DEFAULT_USER_META_POOL_SIZE;
    nvtracker->trackerConfig.gstName = gst_element_get_name(nvtracker);

    nvtracker->trackerIface = NULL;
    nvtracker->output_loop = NULL;
    g_cond_init(&nvtracker->eventCondition);
    g_mutex_init(&nvtracker->eventLock);
}

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean nvtracker_init(GstPlugin *nvtracker)
{
    /* debug category for fltering log messages
     *
     * exchange the string 'Template nvtracker' with your description
     */
    GST_DEBUG_CATEGORY_INIT(gst_nv_tracker_debug, "nvtracker", 0, "nvtracker plugin");

    return gst_element_register(nvtracker, "nvtracker", GST_RANK_PRIMARY, GST_TYPE_NVTRACKER);
}

/* PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "nvtracker"
#endif

/* gstreamer looks for this structure to register nvtrackers
 *
 * exchange the string 'Template nvtracker' with your nvtracker description
 */
GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  nvdsgst_tracker,
                  PACKAGE_DESCRIPTION,
                  nvtracker_init,
                  "6.3",
                  PACKAGE_LICENSE,
                  PACKAGE_NAME,
                  PACKAGE_URL)
