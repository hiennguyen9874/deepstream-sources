#ifndef __GST_NVDSAUDIOTEMPLATE_META_H__
#define __GST_NVDSAUDIOTEMPLATE_META_H__

#include <gst/gst.h>

#include <iostream>

GType gst_mpeg_video_meta_api_get_type(void);
#define GST_AUDIO_TEMPLATE_META_API_TYPE (gst_audio_template_meta_api_get_type())
#define GST_AUDIO_TEMPLATE_META_INFO (gst_audio_template_meta_get_info())

typedef struct _GstAudioTemplateMeta {
    GstMeta meta;
    guint frame_count;
    guint custom_metadata_type;
    guint custom_metadata_size;
    void *custom_metadata_ptr;
} GstAudioTemplateMeta;

GType gst_audio_template_meta_api_get_type(void);
const GstMetaInfo *gst_audio_template_meta_get_info(void);

#endif /*__GST_NVDSAUDIOTEMPLATE_META_H__*/
