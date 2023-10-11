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

bool nvds_rest_roi_parse(const Json::Value &in, NvDsRoiInfo *roi_info)
{
    for (Json::ValueConstIterator it = in.begin(); it != in.end(); ++it) {
        std::string root_val = it.key().asString().c_str();
        roi_info->root_key = root_val;

        const Json::Value sub_root_val = in[root_val]; // object values of root_key

        roi_info->stream_id = sub_root_val.get("stream_id", EMPTY_STRING).asString().c_str();
        if (roi_info->stream_id == "") {
            roi_info->roi_log = "stream_id value not parsed correctly";
            roi_info->status = ROI_UPDATE_FAIL;
        }
        roi_info->roi_count = sub_root_val.get("roi_count", 0).asUInt();
        if (roi_info->roi_count == 0) {
            roi_info->roi_log = "roi id is 0";
            roi_info->status = ROI_UPDATE_FAIL;
        }

        const Json::Value roi_arr = sub_root_val.get("roi", EMPTY_STRING);
        if (roi_arr == "") {
            roi_info->roi_log = "roi is empty";
            roi_info->status = ROI_UPDATE_FAIL;
        }

        for (guint i = 0; i < roi_arr.size(); i++) {
            RoiDimension roi_dim;

            g_strlcpy(roi_dim.roi_id, roi_arr[i]["roi_id"].asString().c_str(),
                      sizeof(roi_dim.roi_id));
            roi_dim.left = roi_arr[i]["left"].asUInt();
            roi_dim.top = roi_arr[i]["top"].asUInt();
            roi_dim.width = roi_arr[i]["width"].asUInt();
            roi_dim.height = roi_arr[i]["height"].asUInt();
            roi_info->vect.push_back(roi_dim);
        }
    }

    return true;
}
