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
