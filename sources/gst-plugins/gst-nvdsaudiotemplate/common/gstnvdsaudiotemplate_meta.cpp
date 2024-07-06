#include "gstnvdsaudiotemplate_meta.h"

static gboolean gst_audio_template_meta_init(GstAudioTemplateMeta *audio_template_meta,
                                             gpointer params,
                                             GstBuffer *buffer)
{
    audio_template_meta->custom_metadata_size = 0;
    audio_template_meta->custom_metadata_ptr = NULL;
    return TRUE;
}

static void gst_audio_template_meta_free(GstAudioTemplateMeta *audio_template_meta,
                                         GstBuffer *buffer)
{
    if (audio_template_meta && audio_template_meta->custom_metadata_ptr) {
        // g_print ("freeing %p\n", audio_template_meta->custom_metadata_ptr);
        free(audio_template_meta->custom_metadata_ptr);
        audio_template_meta->custom_metadata_ptr = NULL;
    }
}

static gboolean gst_audio_template_meta_transform(GstBuffer *dest,
                                                  GstMeta *meta,
                                                  GstBuffer *buffer,
                                                  GQuark type,
                                                  gpointer data)
{
    // TODO :
    return TRUE;
}

GType gst_audio_template_meta_api_get_type(void)
{
    static volatile GType type;
    static const gchar *tags[] = {"memory", NULL};

    if (g_once_init_enter(&type)) {
        GType _type = gst_meta_api_type_register("GstAudioTemplateMetaAPI", tags);

        g_once_init_leave(&type, _type);
    }
    return type;
}

const GstMetaInfo *gst_audio_template_meta_get_info(void)
{
    static const GstMetaInfo *audio_template_meta_info = NULL;

    if (g_once_init_enter((GstMetaInfo **)&audio_template_meta_info)) {
        const GstMetaInfo *meta = gst_meta_register(
            GST_AUDIO_TEMPLATE_META_API_TYPE, "GstAudioTemplateMeta", sizeof(GstAudioTemplateMeta),
            (GstMetaInitFunction)gst_audio_template_meta_init,
            (GstMetaFreeFunction)gst_audio_template_meta_free,
            (GstMetaTransformFunction)gst_audio_template_meta_transform);

        g_once_init_leave((GstMetaInfo **)&audio_template_meta_info, (GstMetaInfo *)meta);
    }

    return audio_template_meta_info;
}
