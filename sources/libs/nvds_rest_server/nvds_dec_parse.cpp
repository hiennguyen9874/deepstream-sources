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
#include "nvds_parse.h"
#include "nvds_rest_server.h"

#define EMPTY_STRING ""

bool nvds_rest_dec_parse(const Json::Value &in, NvDsDecInfo *dec_info)
{
    for (Json::ValueConstIterator it = in.begin(); it != in.end(); ++it) {
        std::string root_val = it.key().asString().c_str();
        dec_info->root_key = root_val;

        const Json::Value sub_root_val = in[root_val]; // object values of root_key

        dec_info->stream_id = sub_root_val.get("stream_id", EMPTY_STRING).asString().c_str();
        if (dec_info->dec_flag == DROP_FRAME_INTERVAL) {
            dec_info->drop_frame_interval = sub_root_val.get("drop_frame_interval", 0).asUInt();

            if (dec_info->drop_frame_interval < 0 || dec_info->drop_frame_interval > 30) {
                dec_info->dec_log = "drop_frame_interval value not parsed correctly, Range: 0 - 30";
                dec_info->status = DROP_FRAME_INTERVAL_UPDATE_FAIL;
            }
        }
        if (dec_info->dec_flag == SKIP_FRAMES) {
            dec_info->skip_frames = sub_root_val.get("skip_frames", 0).asUInt();

            if (dec_info->skip_frames < 0 || dec_info->skip_frames > 2) {
                dec_info->dec_log = "skip_frames value not parsed correctly, Range: 0-2";
                dec_info->status = SKIP_FRAMES_UPDATE_FAIL;
            }
        }
        if (dec_info->dec_flag == LOW_LATENCY_MODE) {
            dec_info->low_latency_mode = sub_root_val.get("low_latency_mode", 0).asBool();
        }
    }

    return true;
}
