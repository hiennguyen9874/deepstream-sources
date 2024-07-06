#include <cstring>
#include <iostream>
#include <string>

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"

using std::cout;
using std::endl;

gboolean parse_tracker_yaml(NvDsTrackerConfig *config, gchar *cfg_file_path)
{
    gboolean ret = FALSE;
    YAML::Node configyml = YAML::LoadFile(cfg_file_path);

    config->display_tracking_id = TRUE;
    config->tracking_id_reset_mode = 0;
    config->input_tensor_meta = FALSE;
    config->input_tensor_gie_id = 0;
    config->compute_hw = 0;
    config->user_meta_pool_size = 16;

    for (YAML::const_iterator itr = configyml["tracker"].begin(); itr != configyml["tracker"].end();
         ++itr) {
        std::string paramKey = itr->first.as<std::string>();
        if (paramKey == "enable") {
            config->enable = itr->second.as<gboolean>();
        } else if (paramKey == "tracker-width") {
            config->width = itr->second.as<gint>();
        } else if (paramKey == "tracker-height") {
            config->height = itr->second.as<gint>();
        } else if (paramKey == "gpu-id") {
            config->gpu_id = itr->second.as<guint>();
        } else if (paramKey == "tracker-surface-type") {
            config->tracking_surf_type = itr->second.as<guint>();
        } else if (paramKey == "ll-config-file") {
            std::string temp = itr->second.as<std::string>();
            char *str = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(str, temp.c_str(), 1024);
            config->ll_config_file = (char *)malloc(sizeof(char) * 1024);
            if (!get_absolute_file_path_yaml(cfg_file_path, str, config->ll_config_file)) {
                g_printerr("Error: Could not parse ll-config-file in tracker.\n");
                g_free(str);
                goto done;
            }
            g_free(str);
        } else if (paramKey == "ll-lib-file") {
            std::string temp = itr->second.as<std::string>();
            char *str = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(str, temp.c_str(), 1024);
            config->ll_lib_file = (char *)malloc(sizeof(char) * 1024);
            if (!get_absolute_file_path_yaml(cfg_file_path, str, config->ll_lib_file)) {
                g_printerr("Error: Could not parse ll-lib-file in tracker.\n");
                g_free(str);
                goto done;
            }
            g_free(str);
        } else if (paramKey == "tracking-surface-type") {
            // Diff b/w this and tracking_surf_type
            config->tracking_surface_type = itr->second.as<guint>();
        } else if (paramKey == "display-tracking-id") {
            config->display_tracking_id = itr->second.as<gboolean>();
        } else if (paramKey == "tracking-id-reset-mode") {
            config->tracking_id_reset_mode = itr->second.as<guint>();
        } else if (paramKey == "input-tensor-meta") {
            config->input_tensor_meta = itr->second.as<gboolean>();
        } else if (paramKey == "tensor-meta-gie-id") {
            config->input_tensor_gie_id = itr->second.as<guint>();
        } else if (paramKey == "compute-hw") {
            config->compute_hw = itr->second.as<guint>();
        } else if (paramKey == "user-meta-pool-size") {
            config->user_meta_pool_size = itr->second.as<guint>();
        } else {
            cout << "Unknown key " << paramKey << " for tracker" << endl;
        }
    }

    ret = TRUE;
done:
    if (!ret) {
        cout << __func__ << " failed" << endl;
    }
    return ret;
}
