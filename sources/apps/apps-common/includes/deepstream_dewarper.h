#ifndef __NVGSTDS_DEWARPER_H__
#define __NVGSTDS_DEWARPER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <gst/gst.h>

typedef struct {
    GstElement *bin;
    GstElement *queue;
    GstElement *src_queue;
    GstElement *nvvidconv;
    GstElement *cap_filter;
    GstElement *dewarper_caps_filter;
    GstElement *nvdewarper;
} NvDsDewarperBin;

typedef struct {
    gboolean enable;
    guint gpu_id;
    guint num_out_buffers;
    guint dewarper_dump_frames;
    gchar *config_file;
    guint nvbuf_memory_type;
    guint source_id;
    guint num_surfaces_per_frame;
    guint num_batch_buffers;
} NvDsDewarperConfig;

gboolean create_dewarper_bin(NvDsDewarperConfig *config, NvDsDewarperBin *bin);

#ifdef __cplusplus
}
#endif

#endif
