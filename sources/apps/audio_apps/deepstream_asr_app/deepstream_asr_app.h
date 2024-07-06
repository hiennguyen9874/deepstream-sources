#ifndef __DEEPSTREAM_VC_APP_H_
#define __DEEPSTREAM_VC_APP_H_

#include <glib.h>
#include <gst/gst.h>
#include <stdio.h>
#include <stdlib.h>

#include "deepstream_asr_config_file_parser.h"

#define CHECK_PTR(ptr) \
    if (ptr == NULL) { \
        return -1;     \
    }

typedef struct __StreamCtx {
    gchar *uri;
    guint stream_id;
    guint has_audio;
    guint bus_id;
    GstElement *asr_pipeline;
    int eos_received;
    NvDsAudioConfig audio_config;
    FILE *FP_asr;
} StreamCtx;

typedef struct __AppCtx {
    guint num_sources;
    StreamCtx *sctx;
    NvDsAppConfig app_config;
} AppCtx;

int create_pipeline(AppCtx *appctx, int stream_num, StreamCtx *sctx);
int start_pipeline(int stream_num, StreamCtx *sctx);
int destroy_pipeline(StreamCtx *sctx);

guint get_num_sources(gchar *cfg_file_path);
gboolean parse_config_file(AppCtx *appctx, gchar *config_file);

#endif
