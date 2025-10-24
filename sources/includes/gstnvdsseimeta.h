#ifndef __GST_NVDS_SEI_META_H__
#define __GST_NVDS_SEI_META_H__

#include <gst/gst.h>

#define GST_VIDEO_SEI_META_API_TYPE (gst_video_sei_meta_api_get_type())
#define GST_VIDEO_SEI_META_INFO (gst_video_sei_meta_get_info())

#define GST_USER_SEI_META g_quark_from_static_string("GST.USER.SEI.META")

typedef struct _GstVideoSEIMeta {
    GstMeta meta;
    guint sei_metadata_type;
    guint sei_metadata_size;
    void *sei_metadata_ptr;
} GstVideoSEIMeta;

GType gst_video_sei_meta_api_get_type(void) asm("gst_video_sei_meta_api_get_type");
const GstMetaInfo *gst_video_sei_meta_get_info(void) asm("gst_video_sei_meta_get_info");

GstVideoSEIMeta *gst_buffer_add_video_sei_meta(GstBuffer *buffer);
GstVideoSEIMeta *gst_buffer_get_video_sei_meta(GstBuffer *buffer);

#endif /*__GST_NVDS_SEI_META_H__*/
