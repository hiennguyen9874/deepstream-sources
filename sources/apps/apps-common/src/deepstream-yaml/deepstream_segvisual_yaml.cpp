#include <cstring>
#include <iostream>
#include <string>

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"

using std::cout;
using std::endl;

#define SEG_OUTPUT_WIDTH 1280
#define SEG_OUTPUT_HEIGHT 720

gboolean parse_segvisual_yaml(NvDsSegVisualConfig *config, gchar *cfg_file_path)
{
    gboolean ret = FALSE;

    /** Default values */
    config->height = SEG_OUTPUT_HEIGHT;
    config->width = SEG_OUTPUT_WIDTH;
    config->gpu_id = 0;
    config->max_batch_size = 1;
    config->nvbuf_memory_type = 0;

    YAML::Node configyml = YAML::LoadFile(cfg_file_path);
    for (YAML::const_iterator itr = configyml["segvisual"].begin();
         itr != configyml["segvisual"].end(); ++itr) {
        std::string paramKey = itr->first.as<std::string>();

        if (paramKey == "enable") {
            config->enable = itr->second.as<gboolean>();
        } else if (paramKey == "gpu-id") {
            config->gpu_id = itr->second.as<guint>();
        } else if (paramKey == "batch-size") {
            config->max_batch_size = itr->second.as<guint>();
        } else if (paramKey == "width") {
            config->width = itr->second.as<guint>();
        } else if (paramKey == "height") {
            config->height = itr->second.as<guint>();
        } else if (paramKey == "nvbuf-memory-type") {
            config->nvbuf_memory_type = itr->second.as<guint>();
        } else {
            cout << "[WARNING] Unknown param found in segvisual: " << paramKey << endl;
        }
    }

    ret = TRUE;

    if (!ret) {
        cout << __func__ << " failed" << endl;
    }
    return ret;
}
