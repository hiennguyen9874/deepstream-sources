#ifndef _NVGSTDS_DSANALYTICS_H_
#define _NVGSTDS_DSANALYTICS_H_

#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // Create a bin for the element only if enabled
    gboolean enable;
    guint unique_id;
    // Config file path having properties for the element
    gchar *config_file_path;
} NvDsDsAnalyticsConfig;

// Struct to store references to the bin and elements
typedef struct {
    GstElement *bin;
    GstElement *queue;
    GstElement *elem_dsanalytics;
} NvDsDsAnalyticsBin;

// Function to create the bin and set properties
gboolean create_dsanalytics_bin(NvDsDsAnalyticsConfig *config, NvDsDsAnalyticsBin *bin);

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_DSANALYTICS_H_ */
