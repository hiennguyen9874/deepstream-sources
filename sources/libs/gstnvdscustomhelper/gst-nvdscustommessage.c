/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "gst-nvdscustommessage.h"

#define STREAM_ADD_STRUCT_NAME "stream-add"
#define STREAM_REMOVE_STRUCT_NAME "stream-remove"
#define PIPELINE_EOS_STRUCT_NAME "force-pipeline-eos"

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

gboolean gst_nvmessage_is_force_pipeline_eos(GstMessage *message)
{
    CHECK_MESSAGE_TYPE(message, PIPELINE_EOS_STRUCT_NAME);
}

GstMessage *gst_nvmessage_force_pipeline_eos(GstObject *obj, gboolean force_eos)
{
    GstStructure *str =
        gst_structure_new(PIPELINE_EOS_STRUCT_NAME, "force_eos", G_TYPE_BOOLEAN, force_eos, NULL);

    GstMessage *message = gst_message_new_custom(GST_MESSAGE_ELEMENT, obj, str);

    return message;
}

gboolean gst_nvmessage_parse_force_pipeline_eos(GstMessage *message, gboolean *force_eos)
{
    const GstStructure *str;

    if (!gst_nvmessage_is_force_pipeline_eos(message))
        return FALSE;

    str = gst_message_get_structure(message);

    gst_structure_get_boolean(str, "force_eos", force_eos);
    return TRUE;
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
