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

#include <cuda_runtime_api.h>
#include <glib.h>
#include <gst/gst.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <iostream>

#include "gst-nvevent.h"
#include "gst-nvmessage.h"
#include "gstnvdsmeta.h"
#include "nvds_rest_server.h"
#include "nvds_yml_parser.h"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* By default, OSD process-mode is set to CPU_MODE. To change mode, set as:
 * 1: GPU mode (for Tesla only)
 * 2: HW mode (For Jetson only)
 */
#define OSD_PROCESS_MODE 0

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 0

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 33333

#define MAX_BATCH_SIZE 8

#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

gchar pgie_classes_str[4][32] = {"Vehicle", "TwoWheeler", "Person", "RoadSign"};

static gboolean PERF_MODE = FALSE;

/* tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn tiler_src_pad_buffer_probe(GstPad *pad,
                                                    GstPadProbeInfo *info,
                                                    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    // NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        // int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *)(l_obj->data);
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                person_count++;
                num_rects++;
            }
        }
// Enable below #if to see frame count and object information
#if 0
           g_print ("Frame Number = %d Number of objects = %d "
             "Vehicle Count = %d Person Count = %d\n",
             frame_meta->frame_num, num_rects, vehicle_count, person_count);
#endif

#if 0
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = &display_meta->text_params;
        txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
#endif
    }
    return GST_PAD_PROBE_OK;
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *)data;
    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        g_print("End of stream\n");
        g_main_loop_quit(loop);
        break;
    case GST_MESSAGE_WARNING: {
        gchar *debug;
        GError *error;
        gst_message_parse_warning(msg, &error, &debug);
        g_printerr("WARNING from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
        g_free(debug);
        g_printerr("Warning: %s\n", error->message);
        g_error_free(error);
        break;
    }
    case GST_MESSAGE_ERROR: {
        gchar *debug;
        GError *error;
        gst_message_parse_error(msg, &error, &debug);
        g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
        if (debug)
            g_printerr("Error details: %s\n", debug);
        g_free(debug);
        g_error_free(error);

        g_main_loop_quit(loop);
        break;
    }
    case GST_MESSAGE_ELEMENT: {
        if (gst_nvmessage_is_stream_eos(msg)) {
            guint stream_id;
            if (gst_nvmessage_parse_stream_eos(msg, &stream_id)) {
                g_print("Got EOS from stream %d\n", stream_id);
            }
        }
        break;
    }
    default:
        break;
    }
    return TRUE;
}

int main(int argc, char *argv[])
{
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *multiuribin = NULL, *sink = NULL, *pgie = NULL, *queue1, *queue2,
               *queue3, *queue4, *queue5, *nvvidconv = NULL, *tiler = NULL, *nvdslogger = NULL,
               *preprocess = NULL, *nvosd = NULL;

    GstBus *bus = NULL;
    guint bus_watch_id;
    GstPad *tiler_src_pad = NULL;

    guint tiler_rows, tiler_columns;
    PERF_MODE =
        g_getenv("NVDS_TEST3_PERF_MODE") && !g_strcmp0(g_getenv("NVDS_TEST3_PERF_MODE"), "1");

    gchar *nvdspreprocess_config_file = (gchar *)"config_preprocess.txt";

    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    /* Check input arguments */
    if (argc < 2) {
        g_printerr("Usage: %s <yml file>\n", argv[0]);
        return -1;
    }

    /* Standard GStreamer initialization */
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new("dsserver-pipeline");

    /* Create nvmultiurisrcbin instance to use REST server feature. */
    multiuribin = gst_element_factory_make("nvmultiurisrcbin", "multiuribin");

    if (!pipeline || !multiuribin) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    gst_bin_add(GST_BIN(pipeline), multiuribin);

    GList *src_list = NULL;
    nvds_parse_source_list(&src_list, argv[1], "source-list");

    gchar *uri = (char *)(src_list)->data;

    g_object_set(G_OBJECT(multiuribin), "port", "9000", NULL);
    g_object_set(G_OBJECT(multiuribin), "uri-list", uri, NULL);
    g_object_set(G_OBJECT(multiuribin), "live-source", 1, "width", MUXER_OUTPUT_WIDTH, "height",
                 MUXER_OUTPUT_HEIGHT, "max-batch-size", MAX_BATCH_SIZE, "batched-push-timeout",
                 MUXER_BATCH_TIMEOUT_USEC, NULL);

    g_list_free(src_list);

    preprocess = gst_element_factory_make("nvdspreprocess", "preprocess-plugin");

    g_object_set(G_OBJECT(preprocess), "config-file", nvdspreprocess_config_file, NULL);
    /* Use nvinfer to infer on batched frame. */
    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");

    /* Add queue elements between every two elements */
    queue1 = gst_element_factory_make("queue", "queue1");
    queue2 = gst_element_factory_make("queue", "queue2");
    queue3 = gst_element_factory_make("queue", "queue3");
    queue4 = gst_element_factory_make("queue", "queue4");
    queue5 = gst_element_factory_make("queue", "queue5");

    /* Use nvdslogger for perf measurement. */
    nvdslogger = gst_element_factory_make("nvdslogger", "nvdslogger");

    g_object_set(G_OBJECT(nvdslogger), "fps-measurement-interval-sec", 1, NULL);

    /* Use nvtiler to composite the batched frames into a 2D tiled array based
     * on the source of the frames. */
    tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");

    /* Use convertor to convert from NV12 to RGBA as required by nvosd */
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");

    /* Create OSD to draw on the converted RGBA buffer */
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

    if (PERF_MODE) {
        sink = gst_element_factory_make("fakesink", "nvvideo-renderer");
    } else {
        /* Finally render the osd output */
        if (prop.integrated) {
            sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
        } else {
            sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
        }
    }

    if (!preprocess || !pgie || !nvdslogger || !tiler || !nvvidconv || !nvosd || !sink) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    g_object_set(G_OBJECT(pgie), "config-file-path", "dsserver_pgie_config.yml", NULL);

    g_object_set(G_OBJECT(pgie), "batch-size", MAX_BATCH_SIZE, NULL);

    nvds_parse_osd(nvosd, argv[1], "osd");

    tiler_rows = (guint)sqrt(MAX_BATCH_SIZE);
    tiler_columns = (guint)ceil(1.0 * MAX_BATCH_SIZE / tiler_rows);
    g_object_set(G_OBJECT(tiler), "rows", tiler_rows, "columns", tiler_columns, NULL);

    nvds_parse_tiler(tiler, argv[1], "tiler");
    nvds_parse_egl_sink(sink, argv[1], "sink");

    /* we add a message handler */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    /* Set up the pipeline */
    /* we add all elements into the pipeline */
    gst_bin_add_many(GST_BIN(pipeline), queue1, preprocess, pgie, queue2, nvdslogger, tiler, queue3,
                     nvvidconv, queue4, nvosd, queue5, sink, NULL);
    /* we link the elements together
     * nvmultiurisrcbin -> nvinfer -> nvdslogger -> nvtiler -> nvvidconv -> nvosd
     * -> video-renderer */
    if (!gst_element_link_many(multiuribin, queue1, preprocess, pgie, queue2, nvdslogger, tiler,
                               queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL)) {
        g_printerr("Elements could not be linked. Exiting.\n");
        return -1;
    }

    /* Lets add probe to get informed of the meta data generated, we add probe to
     * the sink pad of the osd element, since by that time, the buffer would have
     * had got all the metadata. */
    tiler_src_pad = gst_element_get_static_pad(pgie, "src");
    if (!tiler_src_pad)
        g_print("Unable to get src pad\n");
    else
        gst_pad_add_probe(tiler_src_pad, GST_PAD_PROBE_TYPE_BUFFER, tiler_src_pad_buffer_probe,
                          NULL, NULL);
    gst_object_unref(tiler_src_pad);

    /* Set the pipeline to "playing" state */
    g_print("Using file: %s\n", argv[1]);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print("Running...\n");

    g_main_loop_run(loop);

    /* Out of the main loop, clean up nicely */
    g_print("Returned, stopping playback\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    g_print("Deleting pipeline\n");
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    return 0;
}
