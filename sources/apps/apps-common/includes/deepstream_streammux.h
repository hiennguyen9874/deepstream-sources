#ifndef _NVGSTDS_STREAMMUX_H_
#define _NVGSTDS_STREAMMUX_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <gst/gst.h>

typedef struct {
    // Struct members to store config / properties for the element
    gint pipeline_width;
    gint pipeline_height;
    gint batch_size;
    gint batched_push_timeout;
    guint gpu_id;
    guint nvbuf_memory_type;
    gboolean live_source;
    gboolean enable_padding;
    gboolean is_parsed;
    gboolean attach_sys_ts_as_ntp;
    gchar *config_file_path;
} NvDsStreammuxConfig;

// Function to create the bin and set properties
gboolean set_streammux_properties(NvDsStreammuxConfig *config, GstElement *streammux);

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_DSEXAMPLE_H_ */
