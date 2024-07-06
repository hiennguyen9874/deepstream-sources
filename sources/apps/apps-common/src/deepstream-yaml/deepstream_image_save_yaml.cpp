#include <cstring>
#include <iostream>
#include <string>

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"

using std::cout;
using std::endl;

gboolean parse_image_save_yaml(NvDsImageSave *config, gchar *cfg_file_path)
{
    gboolean ret = FALSE;
    /*Default Values*/
    config->enable = FALSE;
    config->output_folder_path = NULL;
    config->frame_to_skip_rules_path = NULL;
    config->min_confidence = 0.0;
    config->max_confidence = 1.0;
    config->min_box_width = 1;
    config->min_box_height = 1;
    config->save_image_full_frame = TRUE;
    config->save_image_cropped_object = FALSE;
    config->second_to_skip_interval = 600;

    YAML::Node configyml = YAML::LoadFile(cfg_file_path);
    for (YAML::const_iterator itr = configyml["img-save"].begin();
         itr != configyml["img-save"].end(); ++itr) {
        std::string paramKey = itr->first.as<std::string>();
        if (paramKey == "enable") {
            config->enable = itr->second.as<gboolean>();
        } else if (paramKey == "output-folder-path") {
            std::string temp = itr->second.as<std::string>();
            config->output_folder_path = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(config->output_folder_path, temp.c_str(), 1024);
        } else if (paramKey == "frame-to-skip-rules-path") {
            std::string temp = itr->second.as<std::string>();
            config->frame_to_skip_rules_path = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(config->frame_to_skip_rules_path, temp.c_str(), 1024);
        } else if (paramKey == "save-img-full-frame") {
            config->save_image_full_frame = itr->second.as<gboolean>();
        } else if (paramKey == "save-img-cropped-obj") {
            config->save_image_cropped_object = itr->second.as<gboolean>();
        } else if (paramKey == "second-to-skip-interval") {
            config->second_to_skip_interval = itr->second.as<guint>();
        } else if (paramKey == "min-confidence") {
            config->min_confidence = itr->second.as<gdouble>();
        } else if (paramKey == "max-confidence") {
            config->max_confidence = itr->second.as<gdouble>();
        } else if (paramKey == "min-box-width") {
            config->min_box_width = itr->second.as<guint>();
        } else if (paramKey == "min-box-height") {
            config->min_box_height = itr->second.as<guint>();
        } else {
            cout << "[WARNING] Unknown param found in image-save: " << paramKey << endl;
        }
    }

    ret = TRUE;

    if (!ret) {
        cout << __func__ << " failed" << endl;
    }
    return ret;
}
