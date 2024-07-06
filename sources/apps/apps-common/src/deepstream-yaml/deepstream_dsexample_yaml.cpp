#include <cstring>
#include <iostream>
#include <string>

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"

using std::cout;
using std::endl;

gboolean parse_dsexample_yaml(NvDsDsExampleConfig *config, gchar *cfg_file_path)
{
    gboolean ret = FALSE;
    YAML::Node configyml = YAML::LoadFile(cfg_file_path);

    for (YAML::const_iterator itr = configyml["ds-example"].begin();
         itr != configyml["ds-example"].end(); ++itr) {
        std::string paramKey = itr->first.as<std::string>();
        if (paramKey == "enable") {
            config->enable = itr->second.as<gboolean>();
        } else if (paramKey == "full-frame") {
            config->full_frame = itr->second.as<gboolean>();
        } else if (paramKey == "processing-width") {
            config->processing_width = itr->second.as<gint>();
        } else if (paramKey == "processing-height") {
            config->processing_height = itr->second.as<gint>();
        } else if (paramKey == "blur-objects") {
            config->blur_objects = itr->second.as<gboolean>();
        } else if (paramKey == "unique-id") {
            config->unique_id = itr->second.as<guint>();
        } else if (paramKey == "gpu-id") {
            config->gpu_id = itr->second.as<guint>();
        } else if (paramKey == "nvbuf-memory-type") {
            config->nvbuf_memory_type = itr->second.as<guint>();
        } else {
            cout << "[WARNING] Unknown param found in dsexample: " << paramKey << endl;
        }
    }

    ret = TRUE;

    if (!ret) {
        cout << __func__ << " failed" << endl;
    }
    return ret;
}