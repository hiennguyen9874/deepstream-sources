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
#include <iostream>

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
        dec_info->drop_frame_interval = sub_root_val.get("drop_frame_interval", 0).asUInt();
    }

    return true;
}
