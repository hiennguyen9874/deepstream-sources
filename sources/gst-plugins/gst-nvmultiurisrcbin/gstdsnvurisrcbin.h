/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#ifndef __GST_DS_NVURISRC_BIN_H__
#define __GST_DS_NVURISRC_BIN_H__

#include <gst/video/video.h>

#include "gst-nvdscommonconfig.h"
#include "gst-nvdssr.h"

G_BEGIN_DECLS

enum {
    PROP_0,
    PROP_URI,
    PROP_NUM_EXTRA_SURF,
    PROP_GPU_DEVICE_ID,
    PROP_DEC_SKIP_FRAMES,
    PROP_SOURCE_TYPE,
    PROP_CUDADEC_MEM_TYPE,
    PROP_DROP_FRAME_INTERVAL,
    PROP_RTP_PROTOCOL,
    PROP_FILE_LOOP,
    PROP_SMART_RECORD,
    PROP_SMART_RECORD_DIR_PATH,
    PROP_SMART_RECORD_FILE_PREFIX,
    PROP_SMART_RECORD_VIDEO_CACHE,
    PROP_SMART_RECORD_CACHE,
    PROP_SMART_RECORD_CONTAINER,
    PROP_SMART_RECORD_MODE,
    PROP_SMART_RECORD_DEFAULT_DURATION,
    PROP_SMART_RECORD_STATUS,
    PROP_RTSP_RECONNECT_INTERVAL,
    PROP_LATENCY,
    PROP_SOURCE_ID,
    PROP_UDP_BUFFER_SIZE,
    PROP_DISABLE_PASSTHROUGH,
    PROP_LAST
};

typedef struct _GstDsNvUriSrcBin {
    GstBin bin;

    GstElement *src_elem;
    GstElement *cap_filter;
    GstElement *cap_filter1;
    GstElement *depay;
    GstElement *parser;
    GstElement *dec_que;
    GstElement *decodebin;
    GstElement *tee;
    GstElement *tee_rtsp_pre_decode;
    GstElement *tee_rtsp_post_decode;
    GstElement *fakesink_queue;
    GstElement *fakesink;
    GstElement *nvvidconv;

    GMutex bin_lock;
    struct timeval last_buffer_time;
    struct timeval last_reconnect_time;
    gulong rtspsrc_monitor_probe;
    gboolean reconfiguring;
    gboolean async_state_watch_running;
    guint64 accumulated_base;
    guint64 prev_accumulated_base;
    GstDsNvUriSrcConfig *config;
    NvDsSRContext *recordCtx;

    guint source_watch_id;

    GstElement *adepay;
    GstElement *aqueue;
    GstElement *aparsebin;
    GstElement *atee;
    GstElement *adecodebin;
    GstElement *audio_convert;
    GstElement *audio_resample;

    gboolean video_elem_populated;
    gboolean audio_elem_populated;
} GstDsNvUriSrcBin;

typedef struct _GstDsNvUriSrcBinClass {
    GstBinClass parent_class;

    NvDsSRStatus (*start_sr)(GstDsNvUriSrcBin *,
                             NvDsSRSessionId *sessionId,
                             guint startTime,
                             guint duration,
                             gpointer userData);
    NvDsSRStatus (*stop_sr)(GstDsNvUriSrcBin *, NvDsSRSessionId sessionId);
    NvDsSRStatus (*sr_done)(GstDsNvUriSrcBin *, NvDsSRRecordingInfo *info, gpointer userData);

} GstDsNvUriSrcBinClass;

/* Standard GStreamer boilerplate */
#define GST_TYPE_DS_NVURISRC_BIN (gst_ds_nvurisrc_bin_get_type())
#define GST_DS_NVURISRC_BIN(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_DS_NVURISRC_BIN, GstDsNvUriSrcBin))
#define GST_DS_NVURISRC_BIN_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_DS_NVURISRC_BIN, GstDsNvUriSrcBinClass))
#define GST_DS_NVURISRC_BIN_GET_CLASS(obj) \
    (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DS_NVURISRC_BIN, GstDsNvUriSrcBinClass))
#define GST_IS_DS_NVURISRC_BIN(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_DS_NVURISRC_BIN))
#define GST_IS_DS_NVURISRC_BIN_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_DS_NVURISRC_BIN))
#define GST_DS_NVURISRC_BIN_CAST(obj) ((GstDsNvUriSrcBin *)(obj))

GType gst_ds_nvurisrc_bin_get_type(void);

G_END_DECLS
#endif /* __GST_DS_NVURISRC_BIN_H__ */
