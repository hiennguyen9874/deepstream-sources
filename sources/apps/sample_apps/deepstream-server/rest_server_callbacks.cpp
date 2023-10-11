/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "rest_server_callbacks.h"

void s_appinstance_callback_impl(NvDsAppInstanceInfo *appinstance_info, void *ctx)
{
    AppCtx *serverappctx = (AppCtx *)ctx;
    (void)serverappctx;

    if (!s_force_eos_handle(serverappctx->nvmultiurisrcbinCreator, appinstance_info)) {
        if (appinstance_info->appinstance_flag == QUIT_APP) {
            appinstance_info->app_log = "Unable to handle force-pipeline-eos nvmultiurisrcbin";
            appinstance_info->status = QUIT_FAIL;
        }
    } else {
        if (appinstance_info->appinstance_flag == QUIT_APP) {
            appinstance_info->status = QUIT_SUCCESS;
            g_print("appinstance force quit success\n");
        }
    }

    return;
}

void s_osd_callback_impl(NvDsOsdInfo *osd_info, void *ctx)
{
    AppCtx *serverappctx = (AppCtx *)ctx;
    (void)serverappctx;

    guint sourceId = std::stoi(osd_info->stream_id);

    if (!find_source(serverappctx->nvmultiurisrcbinCreator, sourceId)) {
        if (osd_info->osd_flag == PROCESS_MODE)
            osd_info->status = PROCESS_MODE_UPDATE_FAIL;
    } else {
        GstEvent *nvevent = gst_nvevent_osd_process_mode_update((char *)osd_info->stream_id.c_str(),
                                                                osd_info->process_mode);
        if (!nvevent) {
            osd_info->status = PROCESS_MODE_UPDATE_FAIL;
            osd_info->osd_log = "nv-osd-process-mode-update event creation failed";
        }

        if (!gst_pad_push_event((GstPad *)(gst_nvmultiurisrcbincreator_get_source_pad(
                                    serverappctx->nvmultiurisrcbinCreator)),
                                nvevent)) {
            g_print("[WARN] nv-osd-process-mode-update event not pushed downstream.. !! \n");
            osd_info->status = PROCESS_MODE_UPDATE_FAIL;
            osd_info->osd_log = "nv-osd-process-mode-update event not pushed";
        } else {
            osd_info->status = PROCESS_MODE_UPDATE_SUCCESS;
        }
    }

    return;
}

void s_mux_callback_impl(NvDsMuxInfo *mux_info, void *ctx)
{
    AppCtx *serverappctx = (AppCtx *)ctx;
    (void)serverappctx;

    if (!set_nvuribin_mux_prop(serverappctx->nvmultiurisrcbinCreator, mux_info)) {
        switch (mux_info->mux_flag) {
        case BATCHED_PUSH_TIMEOUT:
            g_print("[WARN] batched-push-timeout update failed .. !! \n");
            mux_info->status = BATCHED_PUSH_TIMEOUT_UPDATE_FAIL;
            break;
        case MAX_LATENCY:
            g_print("[WARN] max-latency update failed .. !! \n");
            mux_info->status = MAX_LATENCY_UPDATE_FAIL;
            break;
        default:
            break;
        }
    } else {
        switch (mux_info->mux_flag) {
        case BATCHED_PUSH_TIMEOUT:
            mux_info->status = BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS;
            break;
        case MAX_LATENCY:
            mux_info->status = MAX_LATENCY_UPDATE_SUCCESS;
            break;
        default:
            break;
        }
    }

    gst_nvmultiurisrcbincreator_sync_children_states(serverappctx->nvmultiurisrcbinCreator);

    return;
}

void s_enc_callback_impl(NvDsEncInfo *enc_info, void *ctx)
{
    AppCtx *serverappctx = (AppCtx *)ctx;
    (void)serverappctx;
    guint sourceId = std::stoi(enc_info->stream_id);
    GstEvent *nvevent = NULL;

    if (!find_source(serverappctx->nvmultiurisrcbinCreator, sourceId)) {
        switch (enc_info->enc_flag) {
        case BITRATE:
            enc_info->status = BITRATE_UPDATE_FAIL;
            break;
        case FORCE_IDR:
            enc_info->status = FORCE_IDR_UPDATE_FAIL;
            break;
        case FORCE_INTRA:
            enc_info->status = FORCE_INTRA_UPDATE_FAIL;
            break;
        case IFRAME_INTERVAL:
            enc_info->status = IFRAME_INTERVAL_UPDATE_FAIL;
            break;
        default:
            break;
        }
    } else {
        switch (enc_info->enc_flag) {
        case BITRATE:
            nvevent = gst_nvevent_enc_bitrate_update((char *)enc_info->stream_id.c_str(),
                                                     enc_info->bitrate);
            /* send nv-enc-bitrate-update event */
            if (!gst_pad_push_event((GstPad *)(gst_nvmultiurisrcbincreator_get_source_pad(
                                        serverappctx->nvmultiurisrcbinCreator)),
                                    nvevent)) {
                g_print(
                    "[WARN] nv-enc-bitrate-update event not pushed downstream.bitrate update "
                    "failed on encoder.. !! \n");
                enc_info->status = BITRATE_UPDATE_FAIL;
            } else {
                enc_info->status = BITRATE_UPDATE_SUCCESS;
            }
            break;
        case FORCE_IDR:
            nvevent =
                gst_nvevent_enc_force_idr((char *)enc_info->stream_id.c_str(), enc_info->force_idr);
            /* send nv-enc-force-idr event */
            if (!gst_pad_push_event((GstPad *)(gst_nvmultiurisrcbincreator_get_source_pad(
                                        serverappctx->nvmultiurisrcbinCreator)),
                                    nvevent)) {
                g_print(
                    "[WARN] nv-enc-force-idr event not pushed downstream.force IDR frame failed on "
                    "encoder .. !! \n");
                enc_info->status = FORCE_IDR_UPDATE_FAIL;
            } else {
                enc_info->status = FORCE_IDR_UPDATE_SUCCESS;
            }
            break;
        case FORCE_INTRA:
            nvevent = gst_nvevent_enc_force_intra((char *)enc_info->stream_id.c_str(),
                                                  enc_info->force_intra);
            /* send nv-enc-force-intra event */
            if (!gst_pad_push_event((GstPad *)(gst_nvmultiurisrcbincreator_get_source_pad(
                                        serverappctx->nvmultiurisrcbinCreator)),
                                    nvevent)) {
                g_print(
                    "[WARN] nv-enc-force-intra event not pushed downstream.force intra frame "
                    "failed on encoder .. !! \n");
                enc_info->status = FORCE_INTRA_UPDATE_FAIL;
            } else {
                enc_info->status = FORCE_INTRA_UPDATE_SUCCESS;
            }
            break;
        case IFRAME_INTERVAL:
            nvevent = gst_nvevent_enc_iframeinterval_update((char *)enc_info->stream_id.c_str(),
                                                            enc_info->iframeinterval);
            /* send nv-enc-iframeinterval-update event */
            if (!gst_pad_push_event((GstPad *)(gst_nvmultiurisrcbincreator_get_source_pad(
                                        serverappctx->nvmultiurisrcbinCreator)),
                                    nvevent)) {
                g_print(
                    "[WARN] nv-enc-iframeinterval-update event not pushed downstream.iframe "
                    "interval update failed on encoder .. !! \n");
                enc_info->status = IFRAME_INTERVAL_UPDATE_FAIL;
            } else {
                enc_info->status = IFRAME_INTERVAL_UPDATE_SUCCESS;
            }
            break;
        default:
            break;
        }
    }

    gst_nvmultiurisrcbincreator_sync_children_states(serverappctx->nvmultiurisrcbinCreator);

    return;
}

void s_conv_callback_impl(NvDsConvInfo *conv_info, void *ctx)
{
    AppCtx *serverappctx = (AppCtx *)ctx;
    (void)serverappctx;

    guint sourceId = std::stoi(conv_info->stream_id);

    if (!set_nvuribin_conv_prop(serverappctx->nvmultiurisrcbinCreator, sourceId, conv_info)) {
        switch (conv_info->conv_flag) {
        case SRC_CROP:
            g_print("[WARN] source-crop update failed .. !! \n");
            conv_info->status = SRC_CROP_UPDATE_FAIL;
            break;
        case DEST_CROP:
            g_print("[WARN] source-crop update failed .. !! \n");
            conv_info->status = DEST_CROP_UPDATE_FAIL;
            break;
        case FLIP_METHOD:
            g_print("[WARN] flip-method update failed .. !! \n");
            conv_info->status = FLIP_METHOD_UPDATE_FAIL;
            break;
        case INTERPOLATION_METHOD:
            g_print("[WARN] interpolation-method update failed .. !! \n");
            conv_info->status = INTERPOLATION_METHOD_UPDATE_FAIL;
            break;
        default:
            break;
        }
    } else {
        switch (conv_info->conv_flag) {
        case SRC_CROP:
            conv_info->status = SRC_CROP_UPDATE_SUCCESS;
            break;
        case DEST_CROP:
            conv_info->status = DEST_CROP_UPDATE_SUCCESS;
            break;
        case FLIP_METHOD:
            conv_info->status = FLIP_METHOD_UPDATE_SUCCESS;
            break;
        case INTERPOLATION_METHOD:
            conv_info->status = INTERPOLATION_METHOD_UPDATE_SUCCESS;
            break;
        default:
            break;
        }
    }

    gst_nvmultiurisrcbincreator_sync_children_states(serverappctx->nvmultiurisrcbinCreator);

    return;
}

void s_inferserver_callback_impl(NvDsInferServerInfo *inferserver_info, void *ctx)
{
    AppCtx *serverappctx = (AppCtx *)ctx;
    (void)serverappctx;

    guint sourceId = std::stoi(inferserver_info->stream_id);

    if (!find_source(serverappctx->nvmultiurisrcbinCreator, sourceId)) {
        if (inferserver_info->inferserver_flag == INFERSERVER_INTERVAL)
            inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
    } else {
        GstEvent *nvevent = gst_nvevent_infer_interval_update(
            (char *)inferserver_info->stream_id.c_str(), inferserver_info->interval);
        /* send nv-infer-interval-update event */
        if (!gst_pad_push_event((GstPad *)(gst_nvmultiurisrcbincreator_get_source_pad(
                                    serverappctx->nvmultiurisrcbinCreator)),
                                nvevent)) {
            g_print(
                "[WARN] nv-infer-interval-update (inferserver) event not pushed downstream.. !! "
                "\n");
            inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
        } else {
            inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_SUCCESS;
        }
        gst_nvmultiurisrcbincreator_sync_children_states(serverappctx->nvmultiurisrcbinCreator);
    }

    return;
}

void s_infer_callback_impl(NvDsInferInfo *infer_info, void *ctx)
{
    AppCtx *serverappctx = (AppCtx *)ctx;
    (void)serverappctx;

    guint sourceId = std::stoi(infer_info->stream_id);

    if (!find_source(serverappctx->nvmultiurisrcbinCreator, sourceId)) {
        infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
    } else {
        GstEvent *nvevent = gst_nvevent_infer_interval_update((char *)infer_info->stream_id.c_str(),
                                                              infer_info->interval);
        /* send nv-infer-interval-update event */
        if (!gst_pad_push_event((GstPad *)(gst_nvmultiurisrcbincreator_get_source_pad(
                                    serverappctx->nvmultiurisrcbinCreator)),
                                nvevent)) {
            g_print("[WARN] nv-infer-interval-update event not pushed downstream.. !! \n");
            infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
        } else {
            infer_info->status = INFER_INTERVAL_UPDATE_SUCCESS;
        }

        gst_nvmultiurisrcbincreator_sync_children_states(serverappctx->nvmultiurisrcbinCreator);
    }

    return;
}

void s_dec_callback_impl(NvDsDecInfo *dec_info, void *ctx)
{
    AppCtx *serverappctx = (AppCtx *)ctx;
    (void)serverappctx;

    guint sourceId = std::stoi(dec_info->stream_id);

    if (!set_nvuribin_dec_prop(serverappctx->nvmultiurisrcbinCreator, sourceId, dec_info)) {
        switch (dec_info->dec_flag) {
        case DROP_FRAME_INTERVAL:
            g_print("[WARN] drop-frame-interval not set on decoder .. !! \n");
            dec_info->status = DROP_FRAME_INTERVAL_UPDATE_FAIL;
            break;
        case SKIP_FRAMES:
            g_print("[WARN] skip-frame not set on decoder .. !! \n");
            dec_info->status = SKIP_FRAMES_UPDATE_FAIL;
            break;
        case LOW_LATENCY_MODE:
            g_print("[WARN] low-latency-mode not set on decoder .. !! \n");
            dec_info->status = LOW_LATENCY_MODE_UPDATE_FAIL;
            break;
        default:
            break;
        }
    } else {
        switch (dec_info->dec_flag) {
        case DROP_FRAME_INTERVAL:
            dec_info->status = DROP_FRAME_INTERVAL_UPDATE_SUCCESS;
            break;
        case SKIP_FRAMES:
            dec_info->status = SKIP_FRAMES_UPDATE_SUCCESS;
            break;
        case LOW_LATENCY_MODE:
            dec_info->status = LOW_LATENCY_MODE_UPDATE_SUCCESS;
            break;
        default:
            break;
        }
    }

    gst_nvmultiurisrcbincreator_sync_children_states(serverappctx->nvmultiurisrcbinCreator);

    return;
}

void s_roi_callback_impl(NvDsRoiInfo *roi_info, void *ctx)
{
    AppCtx *serverappctx = (AppCtx *)ctx;
    (void)serverappctx;

    guint sourceId = std::stoi(roi_info->stream_id);

    if (!find_source(serverappctx->nvmultiurisrcbinCreator, sourceId)) {
        roi_info->status = ROI_UPDATE_FAIL;
    } else {
        RoiDimension roi_dim[roi_info->roi_count];

        for (int i = 0; i < (int)roi_info->roi_count; i++) {
            g_strlcpy(roi_dim[i].roi_id, roi_info->vect[i].roi_id, sizeof(roi_dim[i].roi_id));
            roi_dim[i].left = roi_info->vect[i].left;
            roi_dim[i].top = roi_info->vect[i].top;
            roi_dim[i].width = roi_info->vect[i].width;
            roi_dim[i].height = roi_info->vect[i].height;
        }

        GstEvent *nvevent = gst_nvevent_new_roi_update((char *)roi_info->stream_id.c_str(),
                                                       roi_info->roi_count, roi_dim);
        /* send nv-new_roi_update event */
        if (!gst_pad_push_event((GstPad *)(gst_nvmultiurisrcbincreator_get_source_pad(
                                    serverappctx->nvmultiurisrcbinCreator)),
                                nvevent)) {
            switch (roi_info->roi_flag) {
            case ROI_UPDATE:
                g_print("[WARN] ROI UPDATE event not pushed downstream.. !! \n");
                roi_info->status = ROI_UPDATE_FAIL;
                break;
            default:
                break;
            }
        } else {
            switch (roi_info->roi_flag) {
            case ROI_UPDATE:
                roi_info->status = ROI_UPDATE_SUCCESS;
                break;
            default:
                break;
            }
        }

        gst_nvmultiurisrcbincreator_sync_children_states(serverappctx->nvmultiurisrcbinCreator);
    }
    return;
}

void s_stream_callback_impl(NvDsStreamInfo *stream_info, void *ctx)
{
    AppCtx *serverappctx = (AppCtx *)ctx;
    (void)serverappctx;

    g_print("Inside s_stream_callback_impl callback +++ \n");

    g_mutex_lock(&serverappctx->bincreator_lock);

    /* check stream_info->value_change to identify stream add/remove */
    if (g_strrstr(stream_info->value_change.c_str(), "add")) {
        g_print("stream_info->value_change is stream add \n");

        GstDsNvUriSrcConfig **sourceConfigs = NULL;
        guint numSourceConfigs = 0;
        /** Check if we can accomodate more sources */
        if (gst_nvmultiurisrcbincreator_get_active_sources_list(
                serverappctx->nvmultiurisrcbinCreator, &numSourceConfigs, &sourceConfigs)) {
            gst_nvmultiurisrcbincreator_src_config_list_free(serverappctx->nvmultiurisrcbinCreator,
                                                             numSourceConfigs, sourceConfigs);
            if (numSourceConfigs >= serverappctx->muxConfig.maxBatchSize) {
                g_print(
                    "Failed to add sensor id=[%s]; "
                    "We have [%d] active sources and max-batch-size is configured to [%d]\n",
                    stream_info->value_camera_id.c_str(), numSourceConfigs,
                    serverappctx->muxConfig.maxBatchSize);
                stream_info->status = STREAM_ADD_FAIL;
                g_mutex_unlock(&serverappctx->bincreator_lock);
                return;
            }
        }
        g_print("we can accomodate more sources\n");

        /** Check if sensor id already exist */
        GstDsNvUriSrcConfig *sourceConfig = NULL;
        if ((sourceConfig = gst_nvmultiurisrcbincreator_get_source_config_by_sensorid(
                 serverappctx->nvmultiurisrcbinCreator, stream_info->value_camera_id.c_str()))) {
            g_print("Failed to add sensor id=[%s]; Already added\n",
                    stream_info->value_camera_id.c_str());
            stream_info->status = STREAM_ADD_FAIL;
            g_mutex_unlock(&serverappctx->bincreator_lock);
            gst_nvmultiurisrcbincreator_src_config_free(sourceConfig);
            return;
        }
        g_print("sensor id don't exist \n");

        /** Add the source */
        serverappctx->config.uri = (gchar *)stream_info->value_camera_url.c_str();
        serverappctx->config.sensorId = (gchar *)stream_info->value_camera_id.c_str();
        serverappctx->config.source_id = ++serverappctx->sourceIdCounter;

        g_print("Adding source now \n");
        gboolean ret = gst_nvmultiurisrcbincreator_add_source(serverappctx->nvmultiurisrcbinCreator,
                                                              &serverappctx->config);
        if (ret == FALSE) {
            g_print("Failed to add sensor id=[%s] uri=[%s]\n", serverappctx->config.sensorId,
                    serverappctx->config.uri);
            stream_info->status = STREAM_ADD_FAIL;
        } else {
            g_print("Successfully added sensor id=[%s] uri=[%s]\n", serverappctx->config.sensorId,
                    serverappctx->config.uri);
            stream_info->status = STREAM_ADD_SUCCESS;
        }
        gst_nvmultiurisrcbincreator_sync_children_states(serverappctx->nvmultiurisrcbinCreator);
        /** clean the config place-holders that we change for each source */
        serverappctx->config.uri = NULL;
        serverappctx->config.sensorId = NULL;

    } else if (g_strrstr(stream_info->value_change.c_str(), "remove")) {
        g_print("stream_info->value_change is stream remove \n");
        /* First, find the GstDsNvUriSrcConfig object from nvmultiurisrcbinCreator
           for the provided sensorId and uri */
        GstDsNvUriSrcConfig const *sourceConfig = gst_nvmultiurisrcbincreator_get_source_config(
            serverappctx->nvmultiurisrcbinCreator, stream_info->value_camera_url.c_str(),
            stream_info->value_camera_id.c_str());
        if (sourceConfig) {
            /* Remove the source */
            gboolean ret = gst_nvmultiurisrcbincreator_remove_source(
                serverappctx->nvmultiurisrcbinCreator, sourceConfig->source_id);
            /* NOTE: after call to gst_nvmultiurisrcbincreator_remove_source, sourceConfig will be
             * invalid */
            sourceConfig = NULL;
            if (ret == FALSE) {
                g_print("Failed to remove sensor\n");
                stream_info->status = STREAM_REMOVE_FAIL;
            } else {
                g_print("Successfully removed sensor\n");
                stream_info->status = STREAM_REMOVE_SUCCESS;
                gst_nvmultiurisrcbincreator_sync_children_states(
                    serverappctx->nvmultiurisrcbinCreator);
            }
        } else {
            g_print("No record found; Failed to remove sensor id=[%s] uri=[%s]\n",
                    stream_info->value_camera_id.c_str(), stream_info->value_camera_url.c_str());
        }
    } else {
        g_print("stream_info->value_change string not supported \n");
    }

    g_mutex_unlock(&serverappctx->bincreator_lock);
    g_print("Exiting s_stream_callback_impl callback \n");
    return;
}