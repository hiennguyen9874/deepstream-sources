#ifndef __NVGSTDS_PREPROCESS_H__
#define __NVGSTDS_PREPROCESS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <gst/gst.h>

typedef struct {
    /** create a bin for the element only if enabled */
    gboolean enable;
    /*gie id on which preprocessing is to be done*/
    gint operate_on_gie_id;
    gboolean is_operate_on_gie_id_set;
    /** config file path having properties for preprocess */
    gchar *config_file_path;
} NvDsPreProcessConfig;

typedef struct {
    GstElement *bin;
    GstElement *queue;
    GstElement *preprocess;
} NvDsPreProcessBin;

/**
 * Initialize @ref NvDsPreProcessBin. It creates and adds preprocess and
 * other elements needed for processing to the bin.
 * It also sets properties mentioned in the configuration file under
 * group @ref CONFIG_GROUP_PREPROCESS
 *
 * @param[in] config pointer to infer @ref NvDsPreProcessConfig parsed from
 *            configuration file.
 * @param[in] bin pointer to @ref NvDsPreProcessBin to be filled.
 *
 * @return true if bin created successfully.
 */
gboolean create_preprocess_bin(NvDsPreProcessConfig *config, NvDsPreProcessBin *bin);

#ifdef __cplusplus
}
#endif

#endif