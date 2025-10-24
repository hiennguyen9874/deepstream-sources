#include "yaml_parser.h"

#include <assert.h>
#include <yaml-cpp/yaml.h>

#include <cstring>
#include <iostream>
#include <string>

#include "cuda_runtime_api.h"

using std::cout;
using std::endl;

static gboolean gst_parse_props_yaml(const gchar *cfg_file_path, cfg_params &cfg_params)
{
    gboolean ret = FALSE;
    YAML::Node configyml = YAML::LoadFile(cfg_file_path);
    if (!(configyml.size() > 0)) {
        cout << "Can't open config file (" << cfg_file_path << ")" << endl;
    }
    for (YAML::const_iterator itr = configyml["property"].begin();
         itr != configyml["property"].end(); ++itr) {
        std::string paramKey = itr->first.as<std::string>();
        if (paramKey == "width") {
            cfg_params.m_tensor_width = itr->second.as<unsigned int>();
        } else if (paramKey == "height") {
            cfg_params.m_tensor_height = itr->second.as<unsigned int>();
        } else {
            std::string paramVal = itr->second.as<std::string>();
            printf("not need %s\n", paramVal.c_str());
        }
    }

    ret = TRUE;
done:
    return ret;
}

/* Parse nvinfer config file for context params. Returns FALSE in case of an error. */
gboolean gst_parse_context_params_yaml(const gchar *cfg_file_path, cfg_params &cfg_params)
{
    gboolean ret = FALSE;

    YAML::Node configyml = YAML::LoadFile(cfg_file_path);
    if (!(configyml.size() > 0)) {
        cout << "Can't open config file (" << cfg_file_path << ")" << endl;
    }
    /* 'property' group is mandatory. */
    if (configyml["property"]) {
        if (!gst_parse_props_yaml(cfg_file_path, cfg_params)) {
            g_printerr("Failed to parse group property\n");
            goto done;
        }
    } else {
        g_printerr("Could not find group property\n");
        goto done;
    }
    ret = TRUE;

done:
    if (!ret) {
        g_printerr("** ERROR: <%s:%d>: failed\n", __func__, __LINE__);
    }
    return ret;
}
