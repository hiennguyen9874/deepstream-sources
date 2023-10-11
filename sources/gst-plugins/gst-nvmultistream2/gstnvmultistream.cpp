/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gst/gst.h>

#ifndef PACKAGE
#define PACKAGE "nvmultistream"
#endif

#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA Multistream mux/demux plugin"
#define BINARY_PACKAGE "NVIDIA Multistream Plugins"
#define URL "http://nvidia.com/"

#include "gstnvstreamdemux.h"
#include "gstnvstreammux.h"

gboolean plugin_init(GstPlugin *plugin);

static gboolean plugin_init_2(GstPlugin *plugin)
{
    const gchar *new_mux_str = g_getenv("USE_NEW_NVSTREAMMUX");
    gboolean use_new_mux = !g_strcmp0(new_mux_str, "yes");

#ifndef ENABLE_GST_NVSTREAMMUX_UNIT_TESTS
    if (!use_new_mux) {
        return plugin_init(plugin);
    } else
#endif
    {
        if (!gst_element_register(plugin, "nvstreammux", GST_RANK_PRIMARY, GST_TYPE_NVSTREAMMUX))
            return FALSE;

        if (!gst_element_register(plugin, "nvstreamdemux", GST_RANK_PRIMARY,
                                  GST_TYPE_NVSTREAMDEMUX))
            return FALSE;
    }

    return TRUE;
}

#if 0
/** NOTE: Disabling all static Gst APIs for loading streammux2
 * based on ENV var: USE_NEW_NVSTREAMMUX
 * TODO: Revert https://git-master.nvidia.com/r/#/c/2127642/
 * when we are ready to drop legacy muxer
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_multistream,
    DESCRIPTION, plugin_init, "6.3", LICENSE, BINARY_PACKAGE, URL)
#endif

#ifdef ENABLE_GST_NVSTREAMMUX_UNIT_TESTS
extern "C" gboolean gGstNvMultistream2StaticInit();
gboolean gGstNvMultistream2StaticInit()
{
    return gst_plugin_register_static(GST_VERSION_MAJOR, GST_VERSION_MINOR, "nvdsgst_multistream",
                                      DESCRIPTION, plugin_init_2, "6.3", LICENSE, BINARY_PACKAGE,
                                      PACKAGE, URL);
}
#endif

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  nvdsgst_multistream,
                  DESCRIPTION,
                  plugin_init_2,
                  "6.3",
                  LICENSE,
                  BINARY_PACKAGE,
                  URL)
