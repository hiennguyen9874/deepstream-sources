#ifndef __NVGSTDS_GIE_H__
#define __NVGSTDS_GIE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <gst/gst.h>

#include "deepstream_config.h"
#include "gstnvdsinfer.h"
#include "gstnvdsmeta.h"

typedef enum {
    NV_DS_GIE_PLUGIN_INFER = 0,
    NV_DS_GIE_PLUGIN_INFER_SERVER,
} NvDsGiePluginType;

typedef struct {
    gboolean enable;

    gchar *config_file_path;

    gboolean input_tensor_meta;

    gboolean override_colors;

    gint operate_on_gie_id;
    gboolean is_operate_on_gie_id_set;
    gint operate_on_classes;

    gint num_operate_on_class_ids;
    gint *list_operate_on_class_ids;

    gboolean have_bg_color;
    NvOSD_ColorParams bbox_bg_color;
    NvOSD_ColorParams bbox_border_color;

    GHashTable *bbox_border_color_table;
    GHashTable *bbox_bg_color_table;

    guint batch_size;
    gboolean is_batch_size_set;

    guint interval;
    gboolean is_interval_set;
    guint unique_id;
    gboolean is_unique_id_set;
    guint gpu_id;
    gboolean is_gpu_id_set;
    guint nvbuf_memory_type;
    gchar *model_engine_file_path;

    gchar *audio_transform;
    guint frame_size;
    gboolean is_frame_size_set;
    guint hop_size;
    gboolean is_hop_size_set;
    guint input_audio_rate;

    gchar *label_file_path;
    guint n_labels;
    guint *n_label_outputs;
    gchar ***labels;

    gchar *raw_output_directory;
    gulong file_write_frame_num;

    gchar *tag;

    NvDsGiePluginType plugin_type;
} NvDsGieConfig;

#ifdef __cplusplus
}
#endif

#endif
