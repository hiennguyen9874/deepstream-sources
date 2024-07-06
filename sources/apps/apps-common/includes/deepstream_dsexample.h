#ifndef _NVGSTDS_DSEXAMPLE_H_
#define _NVGSTDS_DSEXAMPLE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <gst/gst.h>

typedef struct {
    // Create a bin for the element only if enabled
    gboolean enable;
    // Struct members to store config / properties for the element
    gboolean full_frame;
    gint processing_width;
    gint processing_height;
    guint unique_id;
    guint gpu_id;
    // For nvvidconv
    guint nvbuf_memory_type;
} NvDsDsExampleConfig;

// Struct to store references to the bin and elements
typedef struct {
    GstElement *bin;
    GstElement *queue;
    GstElement *pre_conv;
    GstElement *elem_dsexample;
} NvDsDsExampleBin;

// Function to create the bin and set properties
gboolean create_dsexample_bin(NvDsDsExampleConfig *config, NvDsDsExampleBin *bin);

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_DSEXAMPLE_H_ */
