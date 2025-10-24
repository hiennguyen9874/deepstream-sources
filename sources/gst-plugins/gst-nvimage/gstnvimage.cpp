#include <gst/gst.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

#include "gstnvimagedec.h"
#include "gstnvimageenc.h"

#ifndef PACKAGE
#define PACKAGE "nvimage"
#endif

#define PACKAGE_DESCRIPTION "nvidia image decoder encoder plugin"
#define PACKAGE_LICENSE "Proprietary"
#define PACKAGE_NAME "GStreamer nVidia Image Decoder Encoder Plugins"
#define PACKAGE_URL "http://nvidia.com/"

static gboolean plugin_init(GstPlugin *plugin)
{
    /* FIXME: Keeping rank of nvimagedec/enc just below nvjpegdec/enc for now,
     * to avoid breaking any existing usecases. Increase the rank later, once
     * nvjpegdec/enc is deprecated. */
    if (!gst_element_register(plugin, "nvimagedec", GST_RANK_PRIMARY + 14, GST_TYPE_NVIMAGE_DEC))
        return FALSE;

    if (!gst_element_register(plugin, "nvimageenc", GST_RANK_PRIMARY + 9, GST_TYPE_NVIMAGE_ENC))
        return FALSE;

    return TRUE;
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  nvdsgst_image,
                  PACKAGE_DESCRIPTION,
                  plugin_init,
                  "8.0",
                  PACKAGE_LICENSE,
                  PACKAGE_NAME,
                  PACKAGE_URL)
