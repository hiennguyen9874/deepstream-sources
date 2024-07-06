#include <cstring>
#include <iostream>
#include <string>

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"

using std::cout;
using std::endl;

gboolean parse_dewarper_yaml(NvDsDewarperConfig *config, gchar *cfg_file_path)
{
    gboolean ret = FALSE;

    YAML::Node configyml = YAML::LoadFile(cfg_file_path);
    for (YAML::const_iterator itr = configyml["dewarper"].begin();
         itr != configyml["dewarper"].end(); ++itr) {
        std::string paramKey = itr->first.as<std::string>();
        if (paramKey == "enable") {
            config->enable = itr->second.as<gboolean>();
        } else if (paramKey == "gpu-id") {
            config->gpu_id = itr->second.as<guint>();
        } else if (paramKey == "source-id") {
            config->source_id = itr->second.as<guint>();
        } else if (paramKey == "num-out-buffers") {
            config->num_out_buffers = itr->second.as<guint>();
        } else if (paramKey == "num-batch-buffers") {
            config->num_batch_buffers = itr->second.as<guint>();
        } else if (paramKey == "config-file") {
            std::string temp = itr->second.as<std::string>();
            char *str = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(str, temp.c_str(), 1023);
            config->config_file = (char *)malloc(sizeof(char) * 1024);
            if (!get_absolute_file_path_yaml(cfg_file_path, str, config->config_file)) {
                g_printerr("Error: Could not parse config-file in dewarper.\n");
                g_free(str);
                goto done;
            }
            g_free(str);
        } else if (paramKey == "nvbuf-memory-type") {
            config->nvbuf_memory_type = itr->second.as<guint>();
        } else if (paramKey == "num-surfaces-per-frame") {
            config->num_surfaces_per_frame = itr->second.as<guint>();
        } else {
            cout << "[WARNING] Unknown param found in dewarper: " << paramKey << endl;
        }
    }

    ret = TRUE;
done:
    if (!ret) {
        cout << __func__ << " failed" << endl;
    }
    return ret;
}
