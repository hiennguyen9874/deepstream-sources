#include "nvds_parse.h"
#include "nvds_rest_server.h"

#define EMPTY_STRING ""

bool nvds_rest_analytics_parse(const Json::Value &in, NvDsServerAnalyticsInfo *analytics_info)
{
    if (analytics_info->uri.find("/api/v1/") != std::string::npos) {
        for (Json::ValueConstIterator it = in.begin(); it != in.end(); ++it) {
            std::string root_val = it.key().asString().c_str();
            analytics_info->root_key = root_val;

            const Json::Value sub_root_val = in[root_val]; // object values of root_key

            if (analytics_info->analytics_flag == RELOAD_CONFIG) {
                try {
                    analytics_info->config_file_path =
                        sub_root_val.get("config_file_path", EMPTY_STRING).asString().c_str();
                } catch (const std::exception &e) {
                    // Error handling: other exceptions
                    analytics_info->analytics_log =
                        "RELOAD_CONFIG_UPDATE_FAIL, error: " + std::string(e.what());
                    analytics_info->status = RELOAD_CONFIG_UPDATE_FAIL;
                    analytics_info->err_info.code = StatusBadRequest;
                    return false;
                }
            }
        }
    } else {
        g_print("Unsupported REST API version\n");
    }

    return true;
}
