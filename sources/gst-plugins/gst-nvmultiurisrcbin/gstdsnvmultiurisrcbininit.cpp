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

#include <gst/gst.h>

#include "gstdsnvmultiurisrcbin.h"

/* Package and library details required for plugin_init */
#define PACKAGE "DeepStream SDK nvmultiurisrcbin Bin"
#define LICENSE "Proprietary"
#define DESCRIPTION "Deepstream SDK nvmultiurisrcbin Bin"
#define BINARY_PACKAGE "Deepstream SDK nvmultiurisrcbin Bin"
#define URL "http://nvidia.com/"

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean nvmultiurisrcbin_plugin_init(GstPlugin *plugin)
{
    if (!gst_element_register(plugin, "nvmultiurisrcbin", GST_RANK_PRIMARY,
                              GST_TYPE_DS_NVMULTIURISRC_BIN))
        return FALSE;

    return TRUE;
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  nvdsgst_nvmultiurisrcbin,
                  DESCRIPTION,
                  nvmultiurisrcbin_plugin_init,
                  "6.2",
                  LICENSE,
                  BINARY_PACKAGE,
                  URL)
