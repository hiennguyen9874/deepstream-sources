#include <gst/gst.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

#include "gstnvdsmetaextract.h"
#include "gstnvdsmetainsert.h"

#define PACKAGE_LICENSE "Proprietary"
#define PACKAGE_NAME "GStreamer NV DS META Data Processor Plugins"
#define PACKAGE_URL "http://nvidia.com/"
#define PACKAGE_DESCRIPTION "DS Elements for META insertion & extraction"

#ifndef PACKAGE
#define PACKAGE "nvdsmetautils"
#endif

static gboolean plugin_init(GstPlugin *plugin)
{
    gboolean ret = TRUE;
    nvds_metainsert_init(plugin);
    nvds_metaextract_init(plugin);
    return ret;
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  nvdsgst_metautils,
                  PACKAGE_DESCRIPTION,
                  plugin_init,
                  "7.0",
                  PACKAGE_LICENSE,
                  PACKAGE_NAME,
                  PACKAGE_URL)
