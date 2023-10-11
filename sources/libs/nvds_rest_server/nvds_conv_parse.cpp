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

bool nvds_rest_conv_parse(const Json::Value &in, NvDsConvInfo *conv_info)
{
    for (Json::ValueConstIterator it = in.begin(); it != in.end(); ++it) {
        std::string root_val = it.key().asString().c_str();
        conv_info->root_key = root_val;

        const Json::Value sub_root_val = in[root_val]; // object values of root_key

        conv_info->stream_id = sub_root_val.get("stream_id", EMPTY_STRING).asString().c_str();
        if (conv_info->stream_id == "") {
            conv_info->conv_log = "stream_id value not parsed correctly";
        }
        if (conv_info->conv_flag == SRC_CROP) {
            conv_info->src_crop = sub_root_val.get("src_crop", EMPTY_STRING).asString().c_str();

            if (conv_info->src_crop == "") {
                conv_info->conv_log =
                    "src_crop value not parsed correctly,  Use string with values of crop location "
                    "to set the property. e.g. 20:20:40:50";
                conv_info->status = SRC_CROP_UPDATE_FAIL;
            }
        }
        if (conv_info->conv_flag == DEST_CROP) {
            conv_info->dest_crop = sub_root_val.get("dest_crop", EMPTY_STRING).asString().c_str();

            if (conv_info->dest_crop == "") {
                conv_info->conv_log =
                    "dest_crop value not parsed correctly, Use string with values of crop location "
                    "to set the property. e.g. 20:20:40:50";
                conv_info->status = DEST_CROP_UPDATE_FAIL;
            }
        }
        if (conv_info->conv_flag == FLIP_METHOD) {
            conv_info->flip_method = sub_root_val.get("flip_method", 0).asUInt();

            if (conv_info->flip_method < 0 || conv_info->flip_method > 7) {
                conv_info->conv_log =
                    "flip_method value not parsed correctly, Enum value range 0-7";
                conv_info->status = FLIP_METHOD_UPDATE_FAIL;
            }
        }
        if (conv_info->conv_flag == INTERPOLATION_METHOD) {
            conv_info->interpolation_method = sub_root_val.get("interpolation_method", 0).asUInt();
            if (conv_info->interpolation_method < 0 || conv_info->interpolation_method > 6) {
                conv_info->conv_log =
                    "interpolation_method value not parsed correctly, Enum value range 0-6";
                conv_info->status = INTERPOLATION_METHOD_UPDATE_FAIL;
            }
        }
    }

    return true;
}
