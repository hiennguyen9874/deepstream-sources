/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include "nvmsgconv.h"

#include <json-glib/json-glib.h>
#include <stdlib.h>
#include <uuid.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "deepstream_schema.h"

using namespace std;

NvDsMsg2pCtx *nvds_msg2p_ctx_create(const gchar *file, NvDsPayloadType type) {
    NvDsMsg2pCtx *ctx = NULL;
    string str;
    bool retVal = true;

    /*
   * Need to parse configuration / CSV files to get static properties of
   * components (e.g. sensor, place etc.) in case of full deepstream schema.
   */
    if (type == NVDS_PAYLOAD_DEEPSTREAM) {
        g_return_val_if_fail(file, NULL);

        ctx = new NvDsMsg2pCtx;
        ctx->privData = create_deepstream_schema_ctx();

        if (g_str_has_suffix(file, ".csv")) {
            retVal = nvds_msg2p_parse_csv(ctx->privData, file);
        } else {
            retVal = nvds_msg2p_parse_key_value(ctx->privData, file);
        }
    } else {
        ctx = new NvDsMsg2pCtx;
        /* If configuration file is provided for minimal schema,
     * parse it for static values.
     */
        if (file) {
            ctx->privData = create_deepstream_schema_ctx();
            retVal = nvds_msg2p_parse_key_value(ctx->privData, file);
        } else {
            ctx->privData = nullptr;
            retVal = true;
        }
    }

    ctx->payloadType = type;

    if (!retVal) {
        cout << "Error in creating instance" << endl;

        if (ctx && ctx->privData)
            destroy_deepstream_schema_ctx(ctx->privData);

        if (ctx) {
            delete ctx;
            ctx = NULL;
        }
    }
    return ctx;
}

void nvds_msg2p_ctx_destroy(NvDsMsg2pCtx *ctx) {
    destroy_deepstream_schema_ctx(ctx->privData);
    ctx->privData = nullptr;
    delete ctx;
}

NvDsPayload **
nvds_msg2p_generate_multiple(NvDsMsg2pCtx *ctx, NvDsEvent *events, guint eventSize,
                             guint *payloadCount) {
    gchar *message = NULL;
    gint len = 0;
    NvDsPayload **payloads = NULL;
    *payloadCount = 0;
    //Set how many payloads are being sent back to the plugin
    payloads = (NvDsPayload **)g_malloc0(sizeof(NvDsPayload *) * 1);

    if (ctx->payloadType == NVDS_PAYLOAD_DEEPSTREAM) {
        message = generate_event_message(ctx->privData, events->metadata);
        if (message) {
            payloads[*payloadCount] = (NvDsPayload *)g_malloc0(sizeof(NvDsPayload));
            len = strlen(message);
            // Remove '\0' character at the end of string and just copy the content.
            payloads[*payloadCount]->payload = g_memdup(message, len);
            payloads[*payloadCount]->payloadSize = len;
            ++(*payloadCount);
            g_free(message);
        }
    } else if (ctx->payloadType == NVDS_PAYLOAD_DEEPSTREAM_MINIMAL) {
        message = generate_event_message_minimal(ctx->privData, events, eventSize);
        if (message) {
            len = strlen(message);
            payloads[*payloadCount] = (NvDsPayload *)g_malloc0(sizeof(NvDsPayload));
            // Remove '\0' character at the end of string and just copy the content.
            payloads[*payloadCount]->payload = g_memdup(message, len);
            payloads[*payloadCount]->payloadSize = len;
            ++(*payloadCount);
            g_free(message);
        }
    } else if (ctx->payloadType == NVDS_PAYLOAD_CUSTOM) {
        payloads[*payloadCount] = (NvDsPayload *)g_malloc0(sizeof(NvDsPayload));
        payloads[*payloadCount]->payload = (gpointer)g_strdup("CUSTOM Schema");
        payloads[*payloadCount]->payloadSize = strlen((char *)payloads[*payloadCount]->payload) + 1;
        ++(*payloadCount);
    } else
        payloads = NULL;

    return payloads;
}

NvDsPayload *
nvds_msg2p_generate(NvDsMsg2pCtx *ctx, NvDsEvent *events, guint size) {
    gchar *message = NULL;
    gint len = 0;
    NvDsPayload *payload = (NvDsPayload *)g_malloc0(sizeof(NvDsPayload));

    if (ctx->payloadType == NVDS_PAYLOAD_DEEPSTREAM) {
        message = generate_event_message(ctx->privData, events->metadata);
        if (message) {
            len = strlen(message);
            // Remove '\0' character at the end of string and just copy the content.
            payload->payload = g_memdup(message, len);
            payload->payloadSize = len;
            g_free(message);
        }
    } else if (ctx->payloadType == NVDS_PAYLOAD_DEEPSTREAM_MINIMAL) {
        message = generate_event_message_minimal(ctx->privData, events, size);
        if (message) {
            len = strlen(message);
            // Remove '\0' character at the end of string and just copy the content.
            payload->payload = g_memdup(message, len);
            payload->payloadSize = len;
            g_free(message);
        }
    } else if (ctx->payloadType == NVDS_PAYLOAD_CUSTOM) {
        payload->payload = (gpointer)g_strdup("CUSTOM Schema");
        payload->payloadSize = strlen((char *)payload->payload) + 1;
    } else
        payload->payload = NULL;

    return payload;
}

NvDsPayload *
nvds_msg2p_generate_new(NvDsMsg2pCtx *ctx, void *metadataInfo) {
    gchar *message = NULL;
    gint len = 0;
    NvDsMsg2pMetaInfo *meta_info = (NvDsMsg2pMetaInfo *)metadataInfo;
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)meta_info->frameMeta;
    NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)meta_info->objMeta;

    NvDsPayload *payload = (NvDsPayload *)g_malloc0(sizeof(NvDsPayload));

    if (ctx->payloadType == NVDS_PAYLOAD_DEEPSTREAM) {
        message = generate_dsmeta_message(ctx->privData, frame_meta, obj_meta);
        if (message) {
            len = strlen(message);
            // Remove '\0' character at the end of string and just copy the content.
            payload->payload = g_memdup(message, len);
            payload->payloadSize = len;
            g_free(message);
        }
    } else if (ctx->payloadType == NVDS_PAYLOAD_DEEPSTREAM_MINIMAL) {
        message = generate_dsmeta_message_minimal(ctx->privData, frame_meta);
        if (message) {
            len = strlen(message);
            // Remove '\0' character at the end of string and just copy the content.
            payload->payload = g_memdup(message, len);
            payload->payloadSize = len;
            g_free(message);
        }
    } else if (ctx->payloadType == NVDS_PAYLOAD_CUSTOM) {
        payload->payload = (gpointer)g_strdup("CUSTOM Schema");
        payload->payloadSize = strlen((char *)payload->payload) + 1;
    } else
        payload->payload = NULL;

    return payload;
}

NvDsPayload **
nvds_msg2p_generate_multiple_new(NvDsMsg2pCtx *ctx, void *metadataInfo, guint *payloadCount) {
    gchar *message = NULL;
    gint len = 0;
    NvDsPayload **payloads = NULL;
    *payloadCount = 0;
    //Set how many payloads are being sent back to the plugin
    payloads = (NvDsPayload **)g_malloc0(sizeof(NvDsPayload *) * 1);

    NvDsMsg2pMetaInfo *meta_info = (NvDsMsg2pMetaInfo *)metadataInfo;
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)meta_info->frameMeta;
    NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)meta_info->objMeta;

    if (ctx->payloadType == NVDS_PAYLOAD_DEEPSTREAM) {
        message = generate_dsmeta_message(ctx->privData, frame_meta, obj_meta);
        if (message) {
            payloads[*payloadCount] = (NvDsPayload *)g_malloc0(sizeof(NvDsPayload));
            len = strlen(message);
            // Remove '\0' character at the end of string and just copy the content.
            payloads[*payloadCount]->payload = g_memdup(message, len);
            payloads[*payloadCount]->payloadSize = len;
            ++(*payloadCount);
            g_free(message);
        }
    } else if (ctx->payloadType == NVDS_PAYLOAD_DEEPSTREAM_MINIMAL) {
        message = generate_dsmeta_message_minimal(ctx->privData, frame_meta);
        if (message) {
            len = strlen(message);
            payloads[*payloadCount] = (NvDsPayload *)g_malloc0(sizeof(NvDsPayload));
            // Remove '\0' character at the end of string and just copy the content.
            payloads[*payloadCount]->payload = g_memdup(message, len);
            payloads[*payloadCount]->payloadSize = len;
            ++(*payloadCount);
            g_free(message);
        }
    } else if (ctx->payloadType == NVDS_PAYLOAD_CUSTOM) {
        payloads[*payloadCount] = (NvDsPayload *)g_malloc0(sizeof(NvDsPayload));
        payloads[*payloadCount]->payload = (gpointer)g_strdup("CUSTOM Schema");
        payloads[*payloadCount]->payloadSize = strlen((char *)payloads[*payloadCount]->payload) + 1;
        ++(*payloadCount);
    } else
        payloads = NULL;

    return payloads;
}

void nvds_msg2p_release(NvDsMsg2pCtx *ctx, NvDsPayload *payload) {
    g_free(payload->payload);
    g_free(payload);
}
