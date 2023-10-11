/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: MIT
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

#include "gst-nvmultiurisrcbincreator.h"

#include <thread>

#include "gst-nvdscustommessage.h"
#include "gst-nvquery.h"
#include "nvds_rest_server.h"

// #define NVMULTIURISRCBIN_CREATOR_DEBUG
#ifndef NVMULTIURISRCBIN_CREATOR_DEBUG
#define LOGD(...)
#else
#define LOGD(fmt, ...) g_print("[DEBUG %s %d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
#endif

typedef struct {
    GstDsNvUriSrcConfig *config;
    GstElement *uribin;
    NvDst_Handle_NvMultiUriSrcCreator apiHandle;
    GstPad *muxSinkPad;
    GstPad *uribin_src_pad;
    gulong probe_eos_handling;
    gulong newPadAddedHandler;
    gulong elemRemovedHandler;
} NvDsUriSourceInfo;

typedef struct {
    // List if N X nvurisrcbin (NvDsUriSourceInfo) instances
    GList *sourceInfoList;
    // Hash of sourceBin instances from sourceId
    GHashTable *sourceInfoHash;
    GstElement *streammux;
    // Parent bin for N X sources -> nvstreammux
    GstElement *nvmultiurisrcbin;

    // mode can either be video or audio
    NvDsMultiUriMode mode;
    guint numOfActiveSources;
    gboolean using_new_mux;

    GstDsNvStreammuxConfig *muxConfig;
    /*lock to protect book keeping data structures:
     * sourceInfoHash, sourceInfoList, numOfActiveSources
     * from possible data race with add/remove/probe_eos_handling API calls
     */
    GMutex lock;

    /* Thread to handle uribin , sourceinfo removal
     * and release nvstreammux sinkpad on EOS.
     */
    GThread *uribin_removal_thread;
    // Queue to hold sourceinfo which would be released on EOS.
    GQueue *remove_uribin_queue;
    GCond remove_uribin_cond;
    gboolean uribin_removal_thread_stop;
    GMutex uribin_removal_lock;
    guint base_index;

} NvMultiUriSrcBinCreator;

#define NVGSTDS_BIN_ADD_GHOST_PAD_NAMED(bin, elem, pad, ghost_pad_name)          \
    do {                                                                         \
        GstPad *gstpad = gst_element_get_static_pad(elem, pad);                  \
        if (gstpad) {                                                            \
            gst_element_add_pad(bin, gst_ghost_pad_new(ghost_pad_name, gstpad)); \
            gst_object_unref(gstpad);                                            \
        }                                                                        \
    } while (0)

#define NVGSTDS_BIN_ADD_GHOST_PAD(bin, elem, pad) \
    NVGSTDS_BIN_ADD_GHOST_PAD_NAMED(bin, elem, pad, pad)

#define NVGSTDS_ELEM_ADD_PROBE(parent_elem, elem, pad, probe_func, probe_type, probe_data)  \
    ({                                                                                      \
        gulong probe_id = 0;                                                                \
        GstPad *gstpad = gst_element_get_static_pad(elem, pad);                             \
        if (!gstpad) {                                                                      \
            GST_ELEMENT_ERROR(parent_elem, RESOURCE, FAILED,                                \
                              ("Could not find '%s' in '%s'", pad, GST_ELEMENT_NAME(elem)), \
                              (NULL));                                                      \
        } else {                                                                            \
            probe_id = gst_pad_add_probe(gstpad, (GstPadProbeType)(probe_type), probe_func, \
                                         probe_data, NULL);                                 \
            gst_object_unref(gstpad);                                                       \
        }                                                                                   \
        probe_id;                                                                           \
    })

static gboolean s_nvmultiurisrcbincreator_link_element_to_streammux_sink_pad(
    NvDsUriSourceInfo *sourceInfo,
    GstElement *streammux,
    GstPad *src_pad,
    gint index);

static void s_nvmultiurisrcbincreator_set_properties_nvuribin(GstElement *element_,
                                                              GstDsNvUriSrcConfig const *config);

static GstDsNvStreammuxConfig *s_nvmultiurisrcbincreator_create_mux_config(
    GstDsNvStreammuxConfig *muxConfig);

static void s_nvmultiurisrcbincreator_destroy_mux_config(GstDsNvStreammuxConfig *muxConfig);

static GstPadProbeReturn s_nvmultiurisrcbincreator_probe_func_eos_handling(GstPad *pad,
                                                                           GstPadProbeInfo *info,
                                                                           gpointer u_data);

static void s_nvmultiurisrcbincreator_remove_source_info(
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator,
    NvDsUriSourceInfo *sourceInfo);

static void s_nvmultiurisrcbincreator_add_source_info_handlers(NvDsUriSourceInfo *sourceInfo);

static void s_nvmultiurisrcbincreator_remove_source_info_handlers(NvDsUriSourceInfo *sourceInfo);

static gboolean s_nvmultiurisrcbincreator_remove_source_impl(
    NvDst_Handle_NvMultiUriSrcCreator apiHandle,
    guint sourceId,
    gboolean forceSourceStateChange);

static gpointer s_uribin_removal_thread(gpointer data);

gint s_get_source_id(NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator);

static GstBus *s_nvmultiurisrcbincreator_get_bus_from_parent(
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator)
{
    GstObject *parent = NULL;
    GstBus *bus = NULL;
    if (!GST_IS_BIN(nvmultiurisrcbinCreator->nvmultiurisrcbin)) {
        return bus;
    }
    if (!(parent = gst_object_get_parent(GST_OBJECT(nvmultiurisrcbinCreator->nvmultiurisrcbin)))) {
    } else {
        bus = gst_element_get_bus((GstElement *)parent);
        gst_object_unref(parent);
    }
    return bus;
}

gboolean find_source(NvDst_Handle_NvMultiUriSrcCreator apiHandle, guint sourceId)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)apiHandle;

    NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *)g_hash_table_lookup(
        nvmultiurisrcbinCreator->sourceInfoHash, sourceId + (gchar *)NULL);

    if (!sourceInfo) {
        // No source found
        g_print("[WARN] No source found .. !! \n");
        return FALSE;
    }
    return TRUE;
}

gboolean s_force_eos_handle(NvDst_Handle_NvMultiUriSrcCreator apiHandle,
                            NvDsAppInstanceInfo *appinstance_info)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)apiHandle;

    g_object_set(G_OBJECT(nvmultiurisrcbinCreator->streammux), "drop-pipeline-eos", FALSE, NULL);
    gboolean force_eos;

    force_eos = appinstance_info->app_quit;

    if (force_eos) {
        GstBus *bus = s_nvmultiurisrcbincreator_get_bus_from_parent(nvmultiurisrcbinCreator);
        if (bus) {
            gst_bus_post(
                bus, gst_nvmessage_force_pipeline_eos(
                         GST_OBJECT(gst_nvmultiurisrcbincreator_get_bin(nvmultiurisrcbinCreator)),
                         force_eos));
            gst_object_unref(bus);
        }
    }

    return TRUE;
}

gboolean set_nvuribin_mux_prop(NvDst_Handle_NvMultiUriSrcCreator apiHandle, NvDsMuxInfo *mux_info)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)apiHandle;

    const gchar *new_mux_str = g_getenv("USE_NEW_NVSTREAMMUX");
    gboolean use_new_mux = !g_strcmp0(new_mux_str, "yes");

    if (mux_info->mux_flag == BATCHED_PUSH_TIMEOUT && !use_new_mux) {
        g_object_set(G_OBJECT(nvmultiurisrcbinCreator->streammux), "batched-push-timeout",
                     mux_info->batched_push_timeout, NULL);
    }
    if (mux_info->mux_flag == MAX_LATENCY) {
        g_object_set(G_OBJECT(nvmultiurisrcbinCreator->streammux), "max-latency",
                     mux_info->max_latency, NULL);
    }

    return TRUE;
}

gboolean set_nvuribin_conv_prop(NvDst_Handle_NvMultiUriSrcCreator apiHandle,
                                guint sourceId,
                                NvDsConvInfo *conv_info)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)apiHandle;

    NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *)g_hash_table_lookup(
        nvmultiurisrcbinCreator->sourceInfoHash, sourceId + (gchar *)NULL);

    if (!sourceInfo) {
        // No source found
        g_print("[WARN] No source found .. !! \n");
        return FALSE;
    }

    GstElement *uribin = sourceInfo->uribin;
    gboolean ret = TRUE;

    GstIterator *iter = NULL;
    GValue value = {0};
    GstElement *elem = NULL;
    gboolean done = FALSE;
    iter = gst_bin_iterate_elements(GST_BIN(uribin));
    GstElementFactory *factory = NULL;

    while (!done) {
        switch (gst_iterator_next(iter, &value)) {
        case GST_ITERATOR_OK:
            elem = (GstElement *)g_value_get_object(&value);
            factory = GST_ELEMENT_GET_CLASS(elem)->elementfactory;

            if (!g_strcmp0(GST_OBJECT_NAME(factory), "nvvideoconvert")) {
                if (conv_info->conv_flag == SRC_CROP) {
                    g_object_set(G_OBJECT(elem), "src-crop", conv_info->src_crop.c_str(), NULL);
                } else if (conv_info->conv_flag == DEST_CROP) {
                    g_object_set(G_OBJECT(elem), "dest-crop", conv_info->dest_crop.c_str(), NULL);
                } else if (conv_info->conv_flag == FLIP_METHOD) {
                    g_object_set(G_OBJECT(elem), "flip-method", conv_info->flip_method, NULL);
                } else if (conv_info->conv_flag == INTERPOLATION_METHOD) {
                    g_object_set(G_OBJECT(elem), "interpolation-method",
                                 conv_info->interpolation_method, NULL);
                }
            }
            g_value_unset(&value);
            break;
        case GST_ITERATOR_RESYNC:
            gst_iterator_resync(iter);
            break;
        case GST_ITERATOR_ERROR:
            GST_WARNING_OBJECT(GST_BIN(uribin), "error in iterating elements");
            done = TRUE;
            ret = FALSE;
            break;
        case GST_ITERATOR_DONE:
            done = TRUE;
            break;
        }
    }
    return ret;
}

gboolean set_nvuribin_dec_prop(NvDst_Handle_NvMultiUriSrcCreator apiHandle,
                               guint sourceId,
                               NvDsDecInfo *dec_info)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)apiHandle;

    NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *)g_hash_table_lookup(
        nvmultiurisrcbinCreator->sourceInfoHash, sourceId + (gchar *)NULL);

    if (!sourceInfo) {
        // No source found
        g_print("[WARN] No source found .. !! \n");
        return FALSE;
    }

    GstElement *uribin = sourceInfo->uribin;
    gboolean ret = TRUE;

    GstIterator *iter = NULL;
    GValue value = {0};
    GstElement *elem = NULL;
    gboolean done = FALSE;
    iter = gst_bin_iterate_elements(GST_BIN(uribin));

    GstIterator *itr2;
    GValue value2 = {0};
    GstElementFactory *factory = NULL;
    while (!done) {
        switch (gst_iterator_next(iter, &value)) {
        case GST_ITERATOR_OK:
            elem = (GstElement *)g_value_get_object(&value);
            factory = GST_ELEMENT_GET_CLASS(elem)->elementfactory;
            if (!g_strcmp0(GST_OBJECT_NAME(factory), "uridecodebin")) {
                /* Elements inside  uridecodebin */
                for (itr2 = gst_bin_iterate_elements(GST_BIN((elem)));
                     gst_iterator_next(itr2, &value2) == GST_ITERATOR_OK; g_value_reset(&value2)) {
                    GstElement *elem2 = (GstElement *)g_value_get_object(&value2);
                    GstElementFactory *factory2 = GST_ELEMENT_GET_CLASS(elem2)->elementfactory;

                    GstIterator *itr3;
                    GValue value3 = {0};

                    if (!g_strcmp0(GST_OBJECT_NAME(factory2), "decodebin")) {
                        for (itr3 = gst_bin_iterate_elements(GST_BIN((elem2)));
                             gst_iterator_next(itr3, &value3) == GST_ITERATOR_OK;
                             g_value_reset(&value3)) {
                            GstElement *elem3 = (GstElement *)g_value_get_object(&value3);
                            GstElementFactory *factory3 =
                                GST_ELEMENT_GET_CLASS(elem3)->elementfactory;

                            if (!g_strcmp0(GST_OBJECT_NAME(factory3), "nvv4l2decoder")) {
                                if (dec_info->dec_flag == DROP_FRAME_INTERVAL) {
                                    g_object_set(G_OBJECT(elem3), "drop-frame-interval",
                                                 dec_info->drop_frame_interval, NULL);
                                } else if (dec_info->dec_flag == SKIP_FRAMES) {
                                    /* TODO: Define skip-frame enum in rest server header */
                                    g_object_set(G_OBJECT(elem3), "skip-frames",
                                                 dec_info->skip_frames, NULL);
                                } else if (dec_info->dec_flag == LOW_LATENCY_MODE) {
                                    g_object_set(G_OBJECT(elem3), "low-latency-mode",
                                                 dec_info->low_latency_mode, NULL);
                                }
                            }
                        }
                    }
                }
            } else if (!g_strcmp0(GST_OBJECT_NAME(factory), "decodebin")) {
                /* Elements inside  decodebin */
                for (itr2 = gst_bin_iterate_elements(GST_BIN((elem)));
                     gst_iterator_next(itr2, &value2) == GST_ITERATOR_OK; g_value_reset(&value2)) {
                    GstElement *elem2 = (GstElement *)g_value_get_object(&value2);
                    GstElementFactory *factory2 = GST_ELEMENT_GET_CLASS(elem2)->elementfactory;

                    if (!g_strcmp0(GST_OBJECT_NAME(factory2), "nvv4l2decoder")) {
                        if (dec_info->dec_flag == DROP_FRAME_INTERVAL) {
                            g_object_set(G_OBJECT(elem2), "drop-frame-interval",
                                         dec_info->drop_frame_interval, NULL);
                        } else if (dec_info->dec_flag == SKIP_FRAMES) {
                            /* TODO: Define skip-frame enum in rest server header */
                            g_object_set(G_OBJECT(elem2), "skip-frames", dec_info->skip_frames,
                                         NULL);
                        } else if (dec_info->dec_flag == LOW_LATENCY_MODE) {
                            g_object_set(G_OBJECT(elem2), "low-latency-mode",
                                         dec_info->low_latency_mode, NULL);
                        }
                    }
                }
            }
            g_value_unset(&value);
            break;
        case GST_ITERATOR_RESYNC:
            gst_iterator_resync(iter);
            break;
        case GST_ITERATOR_ERROR:
            GST_WARNING_OBJECT(GST_BIN(uribin), "error in iterating elements");
            done = TRUE;
            ret = FALSE;
            break;
        case GST_ITERATOR_DONE:
            done = TRUE;
            break;
        }
    }
    return ret;
}

static GstPadProbeReturn src_pad_query_probe(GstPad *pad, GstPadProbeInfo *info, gpointer data)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)data;

    if (info->type & GST_PAD_PROBE_TYPE_QUERY_UPSTREAM) {
        GstQuery *query = GST_QUERY(info->data);
        if (gst_nvquery_is_batch_size(query) && nvmultiurisrcbinCreator->muxConfig) {
            gst_nvquery_batch_size_set(query, nvmultiurisrcbinCreator->muxConfig->maxBatchSize);
            return GST_PAD_PROBE_HANDLED;
        }
    }
    return GST_PAD_PROBE_OK;
}

NvDst_Handle_NvMultiUriSrcCreator gst_nvmultiurisrcbincreator_init(
    gchar *binName,
    NvDsMultiUriMode mode,
    GstDsNvStreammuxConfig *muxConfig)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
        (NvMultiUriSrcBinCreator *)g_malloc0(sizeof(NvMultiUriSrcBinCreator));
    nvmultiurisrcbinCreator->mode = mode;
    nvmultiurisrcbinCreator->numOfActiveSources = 0;
    nvmultiurisrcbinCreator->base_index = 0;
    if (muxConfig) {
        nvmultiurisrcbinCreator->muxConfig = s_nvmultiurisrcbincreator_create_mux_config(muxConfig);
    }

    const gchar *new_mux_str = g_getenv("USE_NEW_NVSTREAMMUX");
    nvmultiurisrcbinCreator->using_new_mux = !g_strcmp0(new_mux_str, "yes");

    /*
     * Create nvstreammux instance
     * Note: nvmultiurisrcbin support only new nvstreammux to take advantage
     * of adaptive batching required for dynamic sensor provisioning
     */
    nvmultiurisrcbinCreator->streammux = gst_element_factory_make("nvstreammux", "src_bin_muxer");

    if (nvmultiurisrcbinCreator->muxConfig) {
        if (!nvmultiurisrcbinCreator->using_new_mux) {
            // set properties exclusive for legacy nvstreammux
            if (!muxConfig->pipeline_width || !muxConfig->pipeline_height) {
                GST_ERROR_OBJECT(nvmultiurisrcbinCreator->streammux,
                                 "[FATAL ERROR] Mandatory fields not set; pipeline will fail; "
                                 "width=%d height=%d\n",
                                 muxConfig->pipeline_width, muxConfig->pipeline_height);
            }
            g_object_set(nvmultiurisrcbinCreator->streammux, "width", muxConfig->pipeline_width,
                         "height", muxConfig->pipeline_height, "batched-push-timeout",
                         muxConfig->batched_push_timeout, "batch-size", muxConfig->maxBatchSize,
                         /* Note: set batch-size to 1 and let add/remove APIs update
                          * This logic cause pipeline slowdown and hence the maxBatchSize
                          * usage.
                          * TODO; Tracked
                          */
                         NULL);
            if (nvmultiurisrcbinCreator->muxConfig->buffer_pool_size) {
                g_object_set(nvmultiurisrcbinCreator->streammux, "buffer-pool-size",
                             nvmultiurisrcbinCreator->muxConfig->buffer_pool_size, NULL);
            }
            if (nvmultiurisrcbinCreator->muxConfig->compute_hw) {
                g_object_set(nvmultiurisrcbinCreator->streammux, "compute-hw",
                             nvmultiurisrcbinCreator->muxConfig->compute_hw, NULL);
            }
            if (nvmultiurisrcbinCreator->muxConfig->interpolation_method) {
                g_object_set(nvmultiurisrcbinCreator->streammux, "interpolation-method",
                             nvmultiurisrcbinCreator->muxConfig->interpolation_method, NULL);
            }
            if (nvmultiurisrcbinCreator->muxConfig->gpu_id) {
                g_object_set(nvmultiurisrcbinCreator->streammux, "gpu-id",
                             nvmultiurisrcbinCreator->muxConfig->gpu_id, NULL);
            }
            g_object_set(nvmultiurisrcbinCreator->streammux, "nvbuf-memory-type",
                         nvmultiurisrcbinCreator->muxConfig->nvbuf_memory_type, NULL);
            if (nvmultiurisrcbinCreator->muxConfig->live_source) {
                g_object_set(nvmultiurisrcbinCreator->streammux, "live-source",
                             nvmultiurisrcbinCreator->muxConfig->live_source, NULL);
            }
            if (nvmultiurisrcbinCreator->muxConfig->enable_padding) {
                g_object_set(nvmultiurisrcbinCreator->streammux, "enable-padding",
                             nvmultiurisrcbinCreator->muxConfig->enable_padding, NULL);
            }

            if (nvmultiurisrcbinCreator->muxConfig->async_process) {
                g_object_set(nvmultiurisrcbinCreator->streammux, "async-process",
                             nvmultiurisrcbinCreator->muxConfig->async_process, NULL);
            }
        } else {
            // using new nvstreammux
            if (nvmultiurisrcbinCreator->muxConfig->config_file_path) {
                g_object_set(nvmultiurisrcbinCreator->streammux, "config-file-path",
                             nvmultiurisrcbinCreator->muxConfig->config_file_path, NULL);
            }
        }

        // common to both

        if (nvmultiurisrcbinCreator->muxConfig->num_surfaces_per_frame) {
            g_object_set(nvmultiurisrcbinCreator->streammux, "num-surfaces-per-frame",
                         nvmultiurisrcbinCreator->muxConfig->num_surfaces_per_frame, NULL);
        }

        g_object_set(nvmultiurisrcbinCreator->streammux, "attach-sys-ts",
                     nvmultiurisrcbinCreator->muxConfig->attach_sys_ts_as_ntp, NULL);

        if (nvmultiurisrcbinCreator->muxConfig->sync_inputs) {
            g_object_set(nvmultiurisrcbinCreator->streammux, "sync-inputs",
                         nvmultiurisrcbinCreator->muxConfig->sync_inputs, NULL);
        }
        if (nvmultiurisrcbinCreator->muxConfig->max_latency) {
            g_object_set(nvmultiurisrcbinCreator->streammux, "max-latency",
                         nvmultiurisrcbinCreator->muxConfig->max_latency, NULL);
        }
        if (nvmultiurisrcbinCreator->muxConfig->frame_num_reset_on_eos) {
            g_object_set(nvmultiurisrcbinCreator->streammux, "frame-num-reset-on-eos",
                         nvmultiurisrcbinCreator->muxConfig->frame_num_reset_on_eos, NULL);
        }
        if (nvmultiurisrcbinCreator->muxConfig->frame_num_reset_on_stream_reset) {
            g_object_set(nvmultiurisrcbinCreator->streammux, "frame-num-reset-on-stream-reset",
                         nvmultiurisrcbinCreator->muxConfig->frame_num_reset_on_stream_reset, NULL);
        }
        if (nvmultiurisrcbinCreator->muxConfig->frame_duration) {
            g_object_set(nvmultiurisrcbinCreator->streammux, "frame-duration",
                         nvmultiurisrcbinCreator->muxConfig->frame_duration, NULL);
        }

        g_object_set(nvmultiurisrcbinCreator->streammux, "drop-pipeline-eos",
                     nvmultiurisrcbinCreator->muxConfig->no_pipeline_eos, NULL);
    }

    nvmultiurisrcbinCreator->sourceInfoHash = g_hash_table_new(NULL, NULL);

    // Create parent bin
    nvmultiurisrcbinCreator->nvmultiurisrcbin = gst_element_factory_make("bin", binName);
    g_object_set(nvmultiurisrcbinCreator->nvmultiurisrcbin, "async-handling", TRUE, NULL);

    if (!nvmultiurisrcbinCreator->nvmultiurisrcbin) {
        GST_WARNING_OBJECT(nvmultiurisrcbinCreator->nvmultiurisrcbin, "bin creation failed\n");
        return (NvDst_Handle_NvMultiUriSrcCreator)NULL;
    }
    // Add nvstreammux to this bin
    gst_bin_add(GST_BIN(nvmultiurisrcbinCreator->nvmultiurisrcbin),
                nvmultiurisrcbinCreator->streammux);

    // Add ghost src pad for the bin
    NVGSTDS_BIN_ADD_GHOST_PAD(nvmultiurisrcbinCreator->nvmultiurisrcbin,
                              nvmultiurisrcbinCreator->streammux, "src");

    NVGSTDS_ELEM_ADD_PROBE(nvmultiurisrcbinCreator->nvmultiurisrcbin,
                           nvmultiurisrcbinCreator->streammux, "src", src_pad_query_probe,
                           GST_PAD_PROBE_TYPE_QUERY_BOTH, nvmultiurisrcbinCreator);

    nvmultiurisrcbinCreator->remove_uribin_queue = g_queue_new();

    g_cond_init(&nvmultiurisrcbinCreator->remove_uribin_cond);
    nvmultiurisrcbinCreator->uribin_removal_thread_stop = FALSE;

    g_mutex_init(&nvmultiurisrcbinCreator->lock);
    g_mutex_init(&nvmultiurisrcbinCreator->uribin_removal_lock);

    nvmultiurisrcbinCreator->uribin_removal_thread =
        g_thread_new("nvmultiurisrcbincreator-uribin-removal-thread", s_uribin_removal_thread,
                     nvmultiurisrcbinCreator);

    return (NvDst_Handle_NvMultiUriSrcCreator)nvmultiurisrcbinCreator;
}

void gst_nvmultiurisrcbincreator_deinit(NvDst_Handle_NvMultiUriSrcCreator apiHandle)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)apiHandle;

    // Remove all remaining sources
    g_mutex_lock(&nvmultiurisrcbinCreator->lock);
    while (nvmultiurisrcbinCreator->sourceInfoList) {
        NvDsUriSourceInfo *sourceInfo =
            (NvDsUriSourceInfo *)(nvmultiurisrcbinCreator->sourceInfoList->data);
        /** nvmultiurisrcbinCreator->sourceInfoList will change in the below API call */
        guint sourceId = sourceInfo->config->source_id;
        g_mutex_unlock(&nvmultiurisrcbinCreator->lock);
        LOGD("removing source %d\n", sourceId);
        gst_nvmultiurisrcbincreator_remove_source(apiHandle, sourceId);
        g_mutex_lock(&nvmultiurisrcbinCreator->lock);
    }

    /* Cannot unref the bin: nvmultiurisrcbinCreator->nvmultiurisrcbin;
     * The floating ref on this bin will be unref'd only when parent
     * pipeline is unref'd
     */

    s_nvmultiurisrcbincreator_destroy_mux_config(nvmultiurisrcbinCreator->muxConfig);

    if (nvmultiurisrcbinCreator->uribin_removal_thread) {
        g_mutex_lock(&nvmultiurisrcbinCreator->uribin_removal_lock);
        nvmultiurisrcbinCreator->uribin_removal_thread_stop = TRUE;
        g_cond_broadcast(&nvmultiurisrcbinCreator->remove_uribin_cond);
        g_mutex_unlock(&nvmultiurisrcbinCreator->uribin_removal_lock);
        g_thread_join(nvmultiurisrcbinCreator->uribin_removal_thread);
    }

    g_cond_clear(&nvmultiurisrcbinCreator->remove_uribin_cond);
    g_queue_free(nvmultiurisrcbinCreator->remove_uribin_queue);
    g_mutex_unlock(&nvmultiurisrcbinCreator->lock);
    g_mutex_clear(&nvmultiurisrcbinCreator->lock);
    g_mutex_clear(&nvmultiurisrcbinCreator->uribin_removal_lock);
    g_free(nvmultiurisrcbinCreator);
}

static gpointer s_uribin_removal_thread(gpointer data)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)data;

    g_mutex_lock(&nvmultiurisrcbinCreator->uribin_removal_lock);

    while (!nvmultiurisrcbinCreator->uribin_removal_thread_stop) {
        if (g_queue_is_empty(nvmultiurisrcbinCreator->remove_uribin_queue))
            g_cond_wait(&nvmultiurisrcbinCreator->remove_uribin_cond,
                        &nvmultiurisrcbinCreator->uribin_removal_lock);

        while (!g_queue_is_empty(nvmultiurisrcbinCreator->remove_uribin_queue)) {
            NvDsUriSourceInfo *sourceInfo =
                (NvDsUriSourceInfo *)g_queue_pop_head(nvmultiurisrcbinCreator->remove_uribin_queue);
            g_mutex_unlock(&nvmultiurisrcbinCreator->uribin_removal_lock);

            if (GST_IS_PAD(sourceInfo->muxSinkPad)) {
                gst_pad_send_event(sourceInfo->muxSinkPad, gst_event_new_flush_stop(FALSE));
                gst_element_release_request_pad(nvmultiurisrcbinCreator->streammux,
                                                sourceInfo->muxSinkPad);
                gst_object_unref(sourceInfo->muxSinkPad);
            }

            g_object_ref(sourceInfo->uribin);
            if ((!gst_bin_remove(GST_BIN(nvmultiurisrcbinCreator->nvmultiurisrcbin),
                                 sourceInfo->uribin))) {
                GST_WARNING_OBJECT(nvmultiurisrcbinCreator->nvmultiurisrcbin,
                                   "Failed to set remove source-id:%u",
                                   sourceInfo->config->source_id);
                return NULL;
            }

            GstStateChangeReturn state_return = GST_STATE_CHANGE_FAILURE;
            if (GST_IS_BIN(sourceInfo->uribin) &&
                (state_return =
                     gst_element_set_state(GST_ELEMENT(sourceInfo->uribin), GST_STATE_NULL)) ==
                    GST_STATE_CHANGE_FAILURE) {
                GST_WARNING_OBJECT(nvmultiurisrcbinCreator->nvmultiurisrcbin,
                                   "Failed to set stop source-id:%u",
                                   sourceInfo->config->source_id);
                return FALSE;
            }
            gst_object_unref(sourceInfo->uribin);

            g_mutex_lock(&nvmultiurisrcbinCreator->lock);

            s_nvmultiurisrcbincreator_remove_source_info_handlers(sourceInfo);
            s_nvmultiurisrcbincreator_remove_source_info(nvmultiurisrcbinCreator, sourceInfo);
            g_mutex_unlock(&nvmultiurisrcbinCreator->lock);
            g_mutex_lock(&nvmultiurisrcbinCreator->uribin_removal_lock);
        }
    }

    g_mutex_unlock(&nvmultiurisrcbinCreator->uribin_removal_lock);
    return NULL;
}

static GstPadProbeReturn s_nvmultiurisrcbincreator_probe_func_eos_handling(GstPad *pad,
                                                                           GstPadProbeInfo *info,
                                                                           gpointer u_data)
{
    NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *)u_data;
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
        (NvMultiUriSrcBinCreator *)sourceInfo->apiHandle;

    if (info->type & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
        if (GST_EVENT_TYPE(info->data) == GST_EVENT_EOS) {
            GST_INFO_OBJECT(nvmultiurisrcbinCreator->nvmultiurisrcbin, "EOS for source_id=%d\n",
                            sourceInfo->config->source_id);
            g_mutex_lock(&nvmultiurisrcbinCreator->uribin_removal_lock);

            g_queue_push_tail(nvmultiurisrcbinCreator->remove_uribin_queue, sourceInfo);
            g_cond_broadcast(&nvmultiurisrcbinCreator->remove_uribin_cond);
            g_mutex_unlock(&nvmultiurisrcbinCreator->uribin_removal_lock);
            return GST_PAD_PROBE_REMOVE;
        }
    }

    return GST_PAD_PROBE_OK;
}

static void s_nvmultiurisrcbincreator_cb_newpad(GstElement *decodebin, GstPad *pad, gpointer data)
{
    GstCaps *caps = gst_pad_query_caps(pad, NULL);
    const GstStructure *str = gst_caps_get_structure(caps, 0);
    const gchar *name = gst_structure_get_name(str);
    NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *)data;
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
        (NvMultiUriSrcBinCreator *)sourceInfo->apiHandle;

    GST_DEBUG_OBJECT(nvmultiurisrcbinCreator->nvmultiurisrcbin, "added pad %s -- %s\n", name,
                     GST_PAD_NAME(pad));

    /** Link the pad with nvstreammux sink pad according to the current mode of support */
    if ((nvmultiurisrcbinCreator->mode == NVDS_MULTIURISRCBIN_MODE_VIDEO &&
         !strncmp(GST_PAD_NAME(pad), "vsrc_", 5)) ||
        (nvmultiurisrcbinCreator->mode == NVDS_MULTIURISRCBIN_MODE_AUDIO &&
         !strncmp(GST_PAD_NAME(pad), "asrc_", 5))) {
        // Get sink_%d pad from nvstreammux and link to it
        s_nvmultiurisrcbincreator_link_element_to_streammux_sink_pad(
            sourceInfo, nvmultiurisrcbinCreator->streammux, pad, sourceInfo->config->source_id);
        // Attach a probe for EOS handling
        sourceInfo->uribin_src_pad = pad;
        sourceInfo->probe_eos_handling =
            gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                              s_nvmultiurisrcbincreator_probe_func_eos_handling, data, NULL);
    }
#if 0
  /** Link the unnecessary pad with fakesink according to the current mode of support
   * TODO; Leaving this hanging to avoid hang in pipeline teardown
   * Test Case: test_nvdshelper_multiurisrcbin_dynamic_add_remove_video_uri
   */
  if ((nvmultiurisrcbinCreator->mode == NVDS_MULTIURISRCBIN_MODE_VIDEO
          && !strncmp (GST_PAD_NAME (pad), "asrc_", 5))
      || (nvmultiurisrcbinCreator->mode == NVDS_MULTIURISRCBIN_MODE_AUDIO
          && !strncmp (GST_PAD_NAME (pad), "vsrc_", 5))) {
    //Create fakesink, add to bin and link to it
    GstElement *queue = gst_element_factory_make ("queue", NULL);
    GstElement *fakesink = gst_element_factory_make ("fakesink", NULL);
    g_object_set (fakesink, "async", FALSE, "enable-last-sample", FALSE, NULL);
    GstPad *fakesinkSinkPad = gst_element_get_static_pad (queue, "sink");
    gst_bin_add_many (GST_BIN (nvmultiurisrcbinCreator->nvmultiurisrcbin),
        queue, fakesink, NULL);
    gst_pad_link (pad, fakesinkSinkPad);
    gst_object_unref (fakesinkSinkPad);
    gst_element_link (queue, fakesink);
    gst_element_sync_state_with_parent (queue);
    gst_element_sync_state_with_parent (fakesink);
  }
#endif
    gst_nvmultiurisrcbincreator_sync_children_states(sourceInfo->apiHandle);
}

static void s_nvmultiurisrcbincreator_cb_removeelem(GstElement *decodebin,
                                                    GstElement *elem,
                                                    gpointer data)
{
    NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *)data;
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
        (NvMultiUriSrcBinCreator *)sourceInfo->apiHandle;
    GST_DEBUG_OBJECT(nvmultiurisrcbinCreator->nvmultiurisrcbin, "removed element %s\n",
                     GST_ELEMENT_NAME(elem));
    if (strcmp(GST_ELEMENT_NAME(elem), "src") == 0 ||
        strcmp(GST_ELEMENT_NAME(elem), "source") == 0) {
        /** source element is now removed; we expect no more callbacks for this source
         * from GStreamer; also the user did not call the API:
         * gst_nvmultiurisrcbincreator_remove_source() If user did call, this remove-elem callback
         * would have been disconnected So, remove the source info from internal book-keeping here:
         */
        g_mutex_lock(&nvmultiurisrcbinCreator->lock);
        // remove handlers to ensure no more callbacks to handle remove stream
        s_nvmultiurisrcbincreator_remove_source_info_handlers(sourceInfo);
        s_nvmultiurisrcbincreator_remove_source_info(nvmultiurisrcbinCreator, sourceInfo);
        g_mutex_unlock(&nvmultiurisrcbinCreator->lock);
    }
}

GstDsNvUriSrcConfig *gst_nvmultiurisrcbincreator_src_config_dup(GstDsNvUriSrcConfig *sourceConfig)
{
    GstDsNvUriSrcConfig *config = (GstDsNvUriSrcConfig *)g_malloc0(sizeof(GstDsNvUriSrcConfig));
    *config = *sourceConfig;
    /** allocate memory for pointers and copy them over */
    config->uri = g_strdup(sourceConfig->uri);
    config->sensorId = sourceConfig->sensorId ? g_strdup(sourceConfig->sensorId) : NULL;
    config->smart_rec_dir_path = g_strdup(sourceConfig->smart_rec_dir_path);
    config->smart_rec_file_prefix = g_strdup(sourceConfig->smart_rec_file_prefix);
    return config;
}

void gst_nvmultiurisrcbincreator_src_config_free(GstDsNvUriSrcConfig *config)
{
    if (config->uri) {
        g_free(config->uri);
    }
    if (config->sensorId) {
        g_free(config->sensorId);
    }
    if (config->smart_rec_dir_path) {
        g_free(config->smart_rec_dir_path);
    }
    if (config->smart_rec_file_prefix) {
        g_free(config->smart_rec_file_prefix);
    }
    if (config)
        g_free(config);
}

static NvDsUriSourceInfo *s_nvmultiurisrcbincreator_create_source_info(
    GstDsNvUriSrcConfig *sourceConfig,
    NvDst_Handle_NvMultiUriSrcCreator apiHandle)
{
    NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *)g_malloc0(sizeof(NvDsUriSourceInfo));

    /** Copy the source config */
    sourceInfo->config = gst_nvmultiurisrcbincreator_src_config_dup(sourceConfig);

    sourceInfo->apiHandle = apiHandle;
    return sourceInfo;
}

static void s_nvmultiurisrcbincreator_destroy_source_info(NvDsUriSourceInfo *sourceInfo)
{
    gst_nvmultiurisrcbincreator_src_config_free(sourceInfo->config);
    g_free(sourceInfo);
}

static GstDsNvStreammuxConfig *s_nvmultiurisrcbincreator_create_mux_config(
    GstDsNvStreammuxConfig *muxConfig)
{
    GstDsNvStreammuxConfig *config =
        (GstDsNvStreammuxConfig *)g_malloc0(sizeof(GstDsNvStreammuxConfig));
    *config = *muxConfig;
    config->config_file_path = g_strdup(muxConfig->config_file_path);

    return config;
}

static void s_nvmultiurisrcbincreator_destroy_mux_config(GstDsNvStreammuxConfig *muxConfig)
{
    if (muxConfig->config_file_path) {
        g_free(muxConfig->config_file_path);
    }
    if (muxConfig) {
        g_free(muxConfig);
    }
}

gint s_get_source_id(NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator)
{
    gchar pad_name[64];
    for (int i = 0; i < 1024; i++) {
        guint pad_indx = (nvmultiurisrcbinCreator->base_index + i) % 1024;
        g_snprintf(pad_name, sizeof(pad_name), "sink_%u", pad_indx);
        GstPad *test_pad = gst_element_get_static_pad(nvmultiurisrcbinCreator->streammux, pad_name);
        if (!test_pad) {
            nvmultiurisrcbinCreator->base_index = pad_indx + 1;
            return pad_indx;
        }
        gst_object_unref(test_pad);
    }
    return -1;
}

gboolean gst_nvmultiurisrcbincreator_add_source(NvDst_Handle_NvMultiUriSrcCreator apiHandle,
                                                GstDsNvUriSrcConfig *sourceConfig)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)apiHandle;
    g_mutex_lock(&nvmultiurisrcbinCreator->lock);
    GstElement *uribin = gst_element_factory_make("nvurisrcbin", NULL);
    if (!uribin) {
        GST_WARNING_OBJECT(nvmultiurisrcbinCreator->nvmultiurisrcbin,
                           "Could not create element 'nvurisrcbin'");
        g_mutex_unlock(&nvmultiurisrcbinCreator->lock);
        return FALSE;
    }

    NvDsUriSourceInfo *sourceInfo =
        s_nvmultiurisrcbincreator_create_source_info(sourceConfig, apiHandle);
    sourceInfo->uribin = (GstElement *)gst_object_ref(uribin);
    sourceConfig->source_id = s_get_source_id(nvmultiurisrcbinCreator);
    sourceInfo->config->source_id = sourceConfig->source_id;
    // set nvurisrcbin properties
    s_nvmultiurisrcbincreator_set_properties_nvuribin(GST_ELEMENT(uribin), sourceConfig);

    // Add sourceInfo to the list and the hashMap
    nvmultiurisrcbinCreator->sourceInfoList =
        g_list_prepend(nvmultiurisrcbinCreator->sourceInfoList, sourceInfo);
    g_hash_table_insert(nvmultiurisrcbinCreator->sourceInfoHash,
                        sourceConfig->source_id + (char *)NULL, sourceInfo);

    // Add nvurisrcbin instance to the bin
    gst_bin_add(GST_BIN(nvmultiurisrcbinCreator->nvmultiurisrcbin), uribin);

    // Do all necessary calls to g_signal_connect()
    s_nvmultiurisrcbincreator_add_source_info_handlers(sourceInfo);

    nvmultiurisrcbinCreator->numOfActiveSources++;

    /** POST nvmessage stream added on the bus */
    {
        NvDsSensorInfo sensorInfo;
        sensorInfo.source_id = sourceConfig->source_id;
        sensorInfo.sensor_id = sourceConfig->sensorId;
        GstBus *bus = s_nvmultiurisrcbincreator_get_bus_from_parent(nvmultiurisrcbinCreator);
        if (bus) {
            gst_bus_post(bus,
                         gst_nvmessage_new_stream_add(
                             GST_OBJECT(nvmultiurisrcbinCreator->nvmultiurisrcbin), &sensorInfo));
            gst_object_unref(bus);
        }
    }

    g_mutex_unlock(&nvmultiurisrcbinCreator->lock);

    return TRUE;
}

gboolean gst_nvmultiurisrcbincreator_sync_children_states(
    NvDst_Handle_NvMultiUriSrcCreator apiHandle)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)apiHandle;
#if 0
  /* Note: set batch-size to 1 and let add/remove APIs update
   * this logic cause pipeline slowdown and hence the maxBatchSize usage
   * TODO; Tracked and hence code commented.
   */
  if (!nvmultiurisrcbinCreator->using_new_mux) {
    g_object_set (nvmultiurisrcbinCreator->streammux, "batch-size",
        nvmultiurisrcbinCreator->numOfActiveSources, NULL);
  }
#endif
    /*
     * Synchronize the state of every child of bin with the state of bin from a
     * different thread:
     */
    gst_element_call_async(nvmultiurisrcbinCreator->nvmultiurisrcbin,
                           (GstElementCallAsyncFunc)gst_bin_sync_children_states, NULL, NULL);
    return TRUE;
}

static void s_nvmultiurisrcbincreator_remove_source_info(
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator,
    NvDsUriSourceInfo *sourceInfo)
{
    NvDsSensorInfo sensorInfoM;
    sensorInfoM.source_id = sourceInfo->config->source_id;
    sensorInfoM.sensor_id = sourceInfo->config->sensorId;

    /** POST nvmessage stream removed on the bus */
    if (GST_IS_BIN(nvmultiurisrcbinCreator->nvmultiurisrcbin)) {
        GstBus *bus = s_nvmultiurisrcbincreator_get_bus_from_parent(nvmultiurisrcbinCreator);
        if (bus) {
            GstMessage *streamRemoveMsg = gst_nvmessage_new_stream_remove(
                GST_OBJECT(nvmultiurisrcbinCreator->nvmultiurisrcbin), &sensorInfoM);
            if (streamRemoveMsg) {
                gst_bus_post(bus, streamRemoveMsg);
                gst_object_unref(bus);
            }
        }
    }
    // remove sourceInfo from the hash map and the list
    g_hash_table_remove(nvmultiurisrcbinCreator->sourceInfoHash,
                        sourceInfo->config->source_id + (gchar *)NULL);
    nvmultiurisrcbinCreator->sourceInfoList =
        g_list_remove(nvmultiurisrcbinCreator->sourceInfoList, sourceInfo);
    // free sourceInfo
    s_nvmultiurisrcbincreator_destroy_source_info(sourceInfo);
    nvmultiurisrcbinCreator->numOfActiveSources--;
}

void s_nvmultiurisrcbincreator_add_source_info_handlers(NvDsUriSourceInfo *sourceInfo)
{
    // connect necessary signals on uribin for linking to nvstreammux
    sourceInfo->newPadAddedHandler =
        g_signal_connect(G_OBJECT(sourceInfo->uribin), "pad-added",
                         G_CALLBACK(s_nvmultiurisrcbincreator_cb_newpad), sourceInfo);
    /** Note: The callback pad-added will add: sourceInfo->probe_eos_handling probe */

    // connect element-removed signal to remove the source in case
    // the stream got removed before an API call to gst_nvmultiurisrcbincreator_remove_source or EOS
    sourceInfo->elemRemovedHandler =
        g_signal_connect(G_OBJECT(sourceInfo->uribin), "element-removed",
                         G_CALLBACK(s_nvmultiurisrcbincreator_cb_removeelem), sourceInfo);
}

void s_nvmultiurisrcbincreator_remove_source_info_handlers(NvDsUriSourceInfo *sourceInfo)
{
    // Remove all handlers to which we gave sourceInfo pointer as callback param
    // Remove Error handling probe; stream is removed over the API call than an error
    if (sourceInfo->elemRemovedHandler) {
        g_signal_handler_disconnect(G_OBJECT(sourceInfo->uribin), sourceInfo->elemRemovedHandler);
    }
    sourceInfo->elemRemovedHandler = 0;

    if (sourceInfo->newPadAddedHandler) {
        g_signal_handler_disconnect(G_OBJECT(sourceInfo->uribin), sourceInfo->newPadAddedHandler);
    }
    sourceInfo->newPadAddedHandler = 0;
}

gboolean gst_nvmultiurisrcbincreator_remove_source(NvDst_Handle_NvMultiUriSrcCreator apiHandle,
                                                   guint sourceId)
{
    return s_nvmultiurisrcbincreator_remove_source_impl(apiHandle, sourceId, TRUE);
}

gboolean gst_nvmultiurisrcbincreator_remove_source_without_forced_state_change(
    NvDst_Handle_NvMultiUriSrcCreator apiHandle,
    guint sourceId)
{
    return s_nvmultiurisrcbincreator_remove_source_impl(apiHandle, sourceId, FALSE);
}

gboolean s_nvmultiurisrcbincreator_remove_source_impl(NvDst_Handle_NvMultiUriSrcCreator apiHandle,
                                                      guint sourceId,
                                                      gboolean forceSourceStateChange)
{
    GstStateChangeReturn state_return = GST_STATE_CHANGE_FAILURE;
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)apiHandle;
    g_mutex_lock(&nvmultiurisrcbinCreator->lock);
    NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *)g_hash_table_lookup(
        nvmultiurisrcbinCreator->sourceInfoHash, sourceId + (gchar *)NULL);

    if (!sourceInfo) {
        // No source found
        g_mutex_unlock(&nvmultiurisrcbinCreator->lock);
        return FALSE;
    }

    LOGD("removing source %d\n", sourceId);
    // Removing handlers from g_signal_connect() calls
    s_nvmultiurisrcbincreator_remove_source_info_handlers(sourceInfo);

    // Removing probes that were added:
    // Remove the EOS handling probe; stream is removed over the API call than EOS
    if (sourceInfo->uribin_src_pad && GST_IS_PAD(sourceInfo->uribin_src_pad)) {
        /** Note: Removing this probe will ensure the forced EOS
         * to not trigger a call to: s_nvmultiurisrcbincreator_probe_func_eos_handling
         */
        if (sourceInfo->probe_eos_handling) {
            gst_pad_remove_probe(sourceInfo->uribin_src_pad, sourceInfo->probe_eos_handling);
        }
        sourceInfo->probe_eos_handling = 0;
        /* Note: we shall not gst_object_unref(sourceInfo->uribin_src_pad);
         * GStreamer warns:
         * Trying to dispose object "a/vsrc_0", but it still has a parent
         * "dsnvurisrcbin1".
         * You need to let the parent manage the object instead of unreffing
         * the object directly.
         */
    }

    if (forceSourceStateChange) {
        GstElement *uribin = sourceInfo->uribin;

        // set uribin state to NULL
        if (GST_IS_BIN(uribin) && (state_return = gst_element_set_state(uribin, GST_STATE_NULL)) ==
                                      GST_STATE_CHANGE_FAILURE) {
            GST_WARNING_OBJECT(nvmultiurisrcbinCreator->nvmultiurisrcbin,
                               "Failed to set stop source-id:%u", sourceId);
            g_mutex_unlock(&nvmultiurisrcbinCreator->lock);
            return FALSE;
        }
        // release nvstreammux sink pad
        if (GST_IS_PAD(sourceInfo->muxSinkPad)) {
            gst_pad_send_event(sourceInfo->muxSinkPad, gst_event_new_flush_stop(FALSE));
            gst_pad_send_event(sourceInfo->muxSinkPad, gst_event_new_eos());
            gst_element_release_request_pad(nvmultiurisrcbinCreator->streammux,
                                            sourceInfo->muxSinkPad);
            gst_object_unref(sourceInfo->muxSinkPad);
        }
        // remove uribin from nvmultiurisrcbin
        if ((state_return == GST_STATE_CHANGE_SUCCESS || state_return == GST_STATE_CHANGE_ASYNC) &&
            !gst_bin_remove(GST_BIN(nvmultiurisrcbinCreator->nvmultiurisrcbin), uribin)) {
            GST_WARNING_OBJECT(nvmultiurisrcbinCreator->nvmultiurisrcbin,
                               "Failed to set remove source-id:%u", sourceId);
            g_mutex_unlock(&nvmultiurisrcbinCreator->lock);
            return FALSE;
        }
    }
    // remove sourceInfo from the hash map and the list, free sourceInfo
    s_nvmultiurisrcbincreator_remove_source_info(nvmultiurisrcbinCreator, sourceInfo);
    LOGD("removed source %d\n", sourceId);
    g_mutex_unlock(&nvmultiurisrcbinCreator->lock);

    return TRUE;
}

GstPad *gst_nvmultiurisrcbincreator_get_source_pad(NvDst_Handle_NvMultiUriSrcCreator apiHandle)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)apiHandle;
    GstPad *srcpad = gst_element_get_static_pad(nvmultiurisrcbinCreator->nvmultiurisrcbin, "src");
    if (!srcpad) {
        GST_WARNING("nvstreammux request src pad failed. Exiting\n");
        return NULL;
    }
    return srcpad;
}

GstElement *gst_nvmultiurisrcbincreator_get_bin(NvDst_Handle_NvMultiUriSrcCreator apiHandle)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)apiHandle;
    return nvmultiurisrcbinCreator->nvmultiurisrcbin;
}

static gboolean s_nvmultiurisrcbincreator_link_element_to_streammux_sink_pad(
    NvDsUriSourceInfo *sourceInfo,
    GstElement *streammux,
    GstPad *src_pad,
    gint index)
{
    gchar pad_name[16];

    if (index >= 0) {
        g_snprintf(pad_name, 16, "sink_%u", index);
        pad_name[15] = '\0';
    } else {
        strcpy(pad_name, "sink_%u");
    }

    sourceInfo->muxSinkPad = gst_element_get_request_pad(streammux, pad_name);
    if (!sourceInfo->muxSinkPad) {
        GST_WARNING("Failed to get sink pad (%d) from streammux\n", sourceInfo->config->source_id);
        return FALSE;
    }

    if (gst_pad_link(src_pad, sourceInfo->muxSinkPad) != GST_PAD_LINK_OK) {
        GST_WARNING("Failed to link '%s' -> '%s'\n", GST_PAD_NAME(src_pad),
                    GST_ELEMENT_NAME(streammux));
        return FALSE;
    }

    return TRUE;
}

GstDsNvUriSrcConfig *gst_nvmultiurisrcbincreator_get_source_config(
    NvDst_Handle_NvMultiUriSrcCreator apiHandle,
    gchar const *uri,
    gchar const *sensorId)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)apiHandle;
    GstDsNvUriSrcConfig *config = NULL;
    g_mutex_lock(&nvmultiurisrcbinCreator->lock);
    /** Go through the list and find it */
    for (GList *node = nvmultiurisrcbinCreator->sourceInfoList; node; node = g_list_next(node)) {
        NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *)(node->data);
        if (sourceInfo->config->uri && sourceInfo->config->sensorId &&
            g_strrstr(sourceInfo->config->uri, uri) &&
            g_strrstr(sourceInfo->config->sensorId, sensorId)) {
            config = gst_nvmultiurisrcbincreator_src_config_dup(sourceInfo->config);
            break;
        }
    }
    g_mutex_unlock(&nvmultiurisrcbinCreator->lock);
    return config;
}

GstDsNvUriSrcConfig *gst_nvmultiurisrcbincreator_get_source_config_by_sensorid(
    NvDst_Handle_NvMultiUriSrcCreator apiHandle,
    gchar const *sensorId)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)apiHandle;
    GstDsNvUriSrcConfig *config = NULL;
    g_mutex_lock(&nvmultiurisrcbinCreator->lock);
    /** Go through the list and find it */
    for (GList *node = nvmultiurisrcbinCreator->sourceInfoList; node; node = g_list_next(node)) {
        NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *)(node->data);
        if (sourceInfo->config->sensorId && g_strrstr(sourceInfo->config->sensorId, sensorId)) {
            config = gst_nvmultiurisrcbincreator_src_config_dup(sourceInfo->config);
            break;
        }
    }
    g_mutex_unlock(&nvmultiurisrcbinCreator->lock);
    return config;
}

static void s_nvmultiurisrcbincreator_set_properties_nvuribin(GstElement *element_,
                                                              GstDsNvUriSrcConfig const *config)
{
    g_object_set(element_, "uri", config->uri, NULL);
    g_object_set(element_, "source-id", (guint)config->source_id, NULL);
    if (config->num_extra_surfaces)
        g_object_set(element_, "num-extra-surfaces", config->num_extra_surfaces, NULL);
    if (config->gpu_id)
        g_object_set(element_, "gpu-id", config->gpu_id, NULL);
    if (config->skip_frames_type)
        g_object_set(element_, "dec-skip-frames", config->skip_frames_type, NULL);
    g_object_set(element_, "type", SOURCE_TYPE_AUTO, NULL); // always set to auto
    g_object_set(element_, "cudadec-memtype", config->cuda_memory_type, NULL);
    if (config->drop_frame_interval)
        g_object_set(element_, "drop-frame-interval", config->drop_frame_interval, NULL);
    if (config->rtp_protocol)
        g_object_set(element_, "select-rtp-protocol", config->rtp_protocol, NULL);
    if (config->loop)
        g_object_set(element_, "file-loop", config->loop, NULL);
    if (config->smart_record)
        g_object_set(element_, "smart-record", config->smart_record, NULL);
    if (config->smart_rec_dir_path)
        g_object_set(element_, "smart-rec-dir-path", config->smart_rec_dir_path, NULL);
    if (config->smart_rec_file_prefix)
        g_object_set(element_, "smart-rec-file-prefix", config->smart_rec_file_prefix, NULL);
    if (config->smart_rec_cache_size)
        g_object_set(element_, "smart-rec-cache", config->smart_rec_cache_size, NULL);
    if (config->smart_rec_mode)
        g_object_set(element_, "smart-rec-mode", config->smart_rec_mode, NULL);
    if (config->smart_rec_container)
        g_object_set(element_, "smart-rec-container", config->smart_rec_container, NULL);
    if (config->smart_rec_def_duration)
        g_object_set(element_, "smart-rec-default-duration", config->smart_rec_def_duration, NULL);
    // g_object_set(element_, "smart-rec-status", config->, NULL); //Not supported
    if (config->rtsp_reconnect_interval_sec)
        g_object_set(element_, "rtsp-reconnect-interval", config->rtsp_reconnect_interval_sec,
                     NULL);
    if (config->latency)
        g_object_set(element_, "latency", config->latency, NULL);
    if (config->udp_buffer_size)
        g_object_set(element_, "udp-buffer-size", config->udp_buffer_size, NULL);
    g_object_set(element_, "disable-passthrough", config->disable_passthrough, NULL);
}

gboolean gst_nvmultiurisrcbincreator_get_active_sources_list(
    NvDst_Handle_NvMultiUriSrcCreator apiHandle,
    guint *count,
    GstDsNvUriSrcConfig ***configs)
{
    NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *)apiHandle;

    if (!nvmultiurisrcbinCreator || !count || !configs) {
        return FALSE;
    }

    *count = 0;
    *configs = NULL;

    g_mutex_lock(&nvmultiurisrcbinCreator->lock);

    if (nvmultiurisrcbinCreator->sourceInfoList) {
        *count = g_list_length(nvmultiurisrcbinCreator->sourceInfoList);
        *configs = (GstDsNvUriSrcConfig **)g_malloc0(sizeof(GstDsNvUriSrcConfig *) * (*count));
        guint i = 0;
        for (GList *node = nvmultiurisrcbinCreator->sourceInfoList; node;
             node = g_list_next(node)) {
            NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *)(node->data);
            (*configs)[i] = gst_nvmultiurisrcbincreator_src_config_dup(sourceInfo->config);
            i++;
        }
    }

    g_mutex_unlock(&nvmultiurisrcbinCreator->lock);

    return TRUE;
}

void gst_nvmultiurisrcbincreator_src_config_list_free(NvDst_Handle_NvMultiUriSrcCreator apiHandle,
                                                      guint count,
                                                      GstDsNvUriSrcConfig **configs)
{
    for (guint i = 0; i < count; i++) {
        gst_nvmultiurisrcbincreator_src_config_free(configs[i]);
    }
    g_free(configs);
}
