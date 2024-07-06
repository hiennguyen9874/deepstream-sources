#include "gst-nvcustommessage.h"

#define STREAM_ADD_STRUCT_NAME "stream-add"
#define STREAM_REMOVE_STRUCT_NAME "stream-remove"

#define CHECK_MESSAGE_TYPE(message, type)                          \
    do {                                                           \
        const GstStructure *str;                                   \
        if (GST_MESSAGE_TYPE(message) != GST_MESSAGE_ELEMENT)      \
            return FALSE;                                          \
        str = gst_message_get_structure(message);                  \
        return (str != NULL) && gst_structure_has_name(str, type); \
    } while (0)

GstMessage *gst_nvmessage_new_stream_add(GstObject *obj, NvDsSensorInfo *sensor_info)
{
    GstStructure *str =
        gst_structure_new(STREAM_ADD_STRUCT_NAME, "source-id", G_TYPE_UINT, sensor_info->source_id,
                          "sensor-id", G_TYPE_STRING, sensor_info->sensor_id, NULL);

    GstMessage *message = gst_message_new_custom(GST_MESSAGE_ELEMENT, obj, str);

    return message;
}

gboolean gst_nvmessage_is_stream_add(GstMessage *message)
{
    CHECK_MESSAGE_TYPE(message, STREAM_ADD_STRUCT_NAME);
}

gboolean gst_nvmessage_parse_stream_add(GstMessage *message, NvDsSensorInfo *sensor_info)
{
    const GstStructure *str;

    if (!gst_nvmessage_is_stream_add(message))
        return FALSE;

    str = gst_message_get_structure(message);
    gst_structure_get_uint(str, "source-id", &sensor_info->source_id);
    sensor_info->sensor_id = gst_structure_get_string(str, "sensor-id");
    return TRUE;
}

GstMessage *gst_nvmessage_new_stream_remove(GstObject *obj, NvDsSensorInfo *sensor_info)
{
    GstStructure *str = gst_structure_new(STREAM_REMOVE_STRUCT_NAME, "source-id", G_TYPE_UINT,
                                          sensor_info->source_id, "sensor-id", G_TYPE_STRING,
                                          sensor_info->sensor_id, NULL);

    GstMessage *message = gst_message_new_custom(GST_MESSAGE_ELEMENT, obj, str);

    return message;
}

gboolean gst_nvmessage_is_stream_remove(GstMessage *message)
{
    CHECK_MESSAGE_TYPE(message, STREAM_REMOVE_STRUCT_NAME);
}

gboolean gst_nvmessage_parse_stream_remove(GstMessage *message, NvDsSensorInfo *sensor_info)
{
    const GstStructure *str;

    if (!gst_nvmessage_is_stream_remove(message))
        return FALSE;

    str = gst_message_get_structure(message);
    gst_structure_get_uint(str, "source-id", &sensor_info->source_id);
    sensor_info->sensor_id = gst_structure_get_string(str, "sensor-id");
    return TRUE;
}
