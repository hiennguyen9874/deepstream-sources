#ifndef __DEEPSTREAM_FEWSHOT_LEARNING_APP_H__
#define __DEEPSTREAM_FEWSHOT_LEARNING_APP_H__

#include <gst/gst.h>

#include "deepstream_config.h"

typedef struct {
    gint anomaly_count;
    gint meta_number;
    struct timespec timespec_first_frame;
    GstClockTime gst_ts_first_frame;
    GMutex lock_stream_rtcp_sr;
    guint32 id;
    gint frameCount;
    GstClockTime last_ntp_time;
} StreamSourceInfo;

typedef struct {
    StreamSourceInfo streams[MAX_SOURCE_BINS];
} TestAppCtx;

#endif /**< __DEEPSTREAM_TEST5_APP_H__ */
