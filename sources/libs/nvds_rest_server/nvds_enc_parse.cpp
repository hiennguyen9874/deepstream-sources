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
#include "nvds_parse.h"
#include "nvds_rest_server.h"

#define EMPTY_STRING ""

bool nvds_rest_enc_parse(const Json::Value &in, NvDsEncInfo *enc_info)
{
    for (Json::ValueConstIterator it = in.begin(); it != in.end(); ++it) {
        std::string root_val = it.key().asString().c_str();
        enc_info->root_key = root_val;

        const Json::Value sub_root_val = in[root_val]; // object values of root_key

        enc_info->stream_id = sub_root_val.get("stream_id", EMPTY_STRING).asString().c_str();
        if (enc_info->enc_flag == BITRATE) {
            enc_info->bitrate = sub_root_val.get("bitrate", 0).asUInt();
            if (enc_info->bitrate < 0 || enc_info->bitrate > UINT_MAX) {
                enc_info->enc_log =
                    "bitrate value not parsed correctly, Unsigned Integer. Range: 0 - 4294967295";
                enc_info->status = BITRATE_UPDATE_FAIL;
            }
        }
        if (enc_info->enc_flag == FORCE_IDR) {
            enc_info->force_idr = sub_root_val.get("force_idr", 0).asBool();
        }
        if (enc_info->enc_flag == FORCE_INTRA) {
            enc_info->force_intra = sub_root_val.get("force_intra", 0).asBool();
        }
        if (enc_info->enc_flag == IFRAME_INTERVAL) {
            enc_info->iframeinterval = sub_root_val.get("iframeinterval", 0).asUInt();
            if (enc_info->iframeinterval > 0 || enc_info->iframeinterval > 4294967295) {
                enc_info->enc_log =
                    "iframeinterval value not parsed correctly, Unsigned Integer Range: 0 - "
                    "4294967295";
                enc_info->status = IFRAME_INTERVAL_UPDATE_FAIL;
            }
        }
    }

    return true;
}
