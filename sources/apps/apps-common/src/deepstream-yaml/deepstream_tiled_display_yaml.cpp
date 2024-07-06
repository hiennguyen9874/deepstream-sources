#include <iostream>
#include <string>

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"

using std::cout;
using std::endl;

gboolean parse_tiled_display_yaml(NvDsTiledDisplayConfig *config, gchar *cfg_file_path)
{
    gboolean ret = FALSE;

    YAML::Node configyml = YAML::LoadFile(cfg_file_path);
    for (YAML::const_iterator itr = configyml["tiled-display"].begin();
         itr != configyml["tiled-display"].end(); ++itr) {
        std::string paramKey = itr->first.as<std::string>();
        if (paramKey == "enable") {
            config->enable = (NvDsTiledDisplayEnable)itr->second.as<int>();
        } else if (paramKey == "rows") {
            config->rows = itr->second.as<guint>();
        } else if (paramKey == "columns") {
            config->columns = itr->second.as<guint>();
        } else if (paramKey == "width") {
            config->width = itr->second.as<guint>();
        } else if (paramKey == "height") {
            config->height = itr->second.as<guint>();
        } else if (paramKey == "gpu-id") {
            config->gpu_id = itr->second.as<guint>();
        } else if (paramKey == "nvbuf-memory-type") {
            config->nvbuf_memory_type = itr->second.as<guint>();
        } else if (paramKey == "compute-hw") {
            config->compute_hw = itr->second.as<guint>();
        } else {
            cout << "[WARNING] Unknown param found in tiled-display: " << paramKey << endl;
        }
    }
    ret = TRUE;

    if (!ret) {
        cout << __func__ << " failed" << endl;
    }
    return ret;
}
