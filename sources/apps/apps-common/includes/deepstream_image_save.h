#ifndef __NVGSTDS_IMAGE_SAVE_H__
#define __NVGSTDS_IMAGE_SAVE_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    gboolean enable;
    gchar *output_folder_path;
    gboolean save_image_full_frame;
    gboolean save_image_cropped_object;
    gchar *frame_to_skip_rules_path;
    guint second_to_skip_interval;
    gdouble min_confidence;
    gdouble max_confidence;
    guint min_box_width;
    guint min_box_height;
} NvDsImageSave;

#ifdef __cplusplus
}
#endif

#endif
