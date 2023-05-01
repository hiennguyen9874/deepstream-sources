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

#include "gst-nvcustomevent.h"

GstEvent *gst_nvevent_new_roi_update(gchar *stream_id, guint roi_count, RoiDimension *roi_dim)
{
    GstStructure *str = gst_structure_new_empty("nv-roi-update");

    gst_structure_set(str, "stream_id", G_TYPE_STRING, stream_id, "roi-count", G_TYPE_UINT,
                      roi_count, NULL);

    for (int i = 0; i < (int)roi_count; i++) {
        char key[128];
        g_snprintf(key, sizeof(key), "roi_id_%d", i);
        gst_structure_set(str, key, G_TYPE_STRING, (char *)roi_dim[i].roi_id, NULL);
        g_snprintf(key, sizeof(key), "left_%d", i);
        gst_structure_set(str, key, G_TYPE_UINT, roi_dim[i].left, NULL);
        g_snprintf(key, sizeof(key), "top_%d", i);
        gst_structure_set(str, key, G_TYPE_UINT, roi_dim[i].top, NULL);
        g_snprintf(key, sizeof(key), "width_%d", i);
        gst_structure_set(str, key, G_TYPE_UINT, roi_dim[i].width, NULL);
        g_snprintf(key, sizeof(key), "height_%d", i);
        gst_structure_set(str, key, G_TYPE_UINT, roi_dim[i].height, NULL);
    }

    return gst_event_new_custom(GST_NVEVENT_ROI_UPDATE, str);
}

GstEvent *gst_nvevent_infer_interval_update(gchar *stream_id, guint interval)
{
    GstStructure *str = gst_structure_new_empty("nv-infer-interval-update");

    gst_structure_set(str, "stream_id", G_TYPE_STRING, stream_id, "interval", G_TYPE_UINT, interval,
                      NULL);

    return gst_event_new_custom(GST_NVEVENT_INFER_INTERVAL_UPDATE, str);
}

void gst_nvevent_parse_roi_update(GstEvent *event,
                                  gchar **stream_id,
                                  guint *roi_count,
                                  RoiDimension **roi_dim)
{
    if ((GstEventType)GST_NVEVENT_ROI_UPDATE == GST_EVENT_TYPE(event)) {
        const GstStructure *str = gst_event_get_structure(event);

        gst_structure_get(str, "stream_id", G_TYPE_STRING, stream_id, "roi-count", G_TYPE_UINT,
                          roi_count, NULL);

        *roi_dim = (RoiDimension *)g_malloc(sizeof(RoiDimension) * (*roi_count));

        for (int i = 0; i < (int)*roi_count; i++) {
            char key[128];
            gchar *roi_id;
            g_snprintf(key, sizeof(key), "roi_id_%d", i);
            gst_structure_get(str, key, G_TYPE_STRING, &(roi_id), NULL);
            g_strlcpy(((*roi_dim)[i].roi_id), roi_id, sizeof(((*roi_dim)[i].roi_id)));
            g_free(roi_id);
            g_snprintf(key, sizeof(key), "left_%d", i);
            gst_structure_get(str, key, G_TYPE_UINT, &((*roi_dim)[i].left), NULL);
            g_snprintf(key, sizeof(key), "top_%d", i);
            gst_structure_get(str, key, G_TYPE_UINT, &((*roi_dim)[i].top), NULL);
            g_snprintf(key, sizeof(key), "width_%d", i);
            gst_structure_get(str, key, G_TYPE_UINT, &((*roi_dim)[i].width), NULL);
            g_snprintf(key, sizeof(key), "height_%d", i);
            gst_structure_get(str, key, G_TYPE_UINT, &((*roi_dim)[i].height), NULL);
        }
    }
}

void gst_nvevent_parse_infer_interval_update(GstEvent *event, gchar **stream_id, guint *interval)
{
    if ((GstEventType)GST_NVEVENT_INFER_INTERVAL_UPDATE == GST_EVENT_TYPE(event)) {
        const GstStructure *str = gst_event_get_structure(event);

        gst_structure_get(str, "stream_id", G_TYPE_STRING, stream_id, "interval", G_TYPE_UINT,
                          interval, NULL);
    }
}
