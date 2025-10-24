#include <iostream>

#include "nvds_parse.h"
#include "nvds_rest_server.h"

bool nvds_rest_nvtracker_parse(const Json::Value &in, NvDsServerNvTrackerInfo *trackerInfo)
{
    if (trackerInfo->uri.find("/api/v1/") != std::string::npos) {
        for (Json::ValueConstIterator it = in.begin(); it != in.end(); ++it) {
            std::string root_val = it.key().asString().c_str();
            trackerInfo->root_key = root_val;

            const Json::Value sub_root_val = in[root_val]; // object values of root_key

            trackerInfo->stream_id = sub_root_val.get("stream_id", "").asString().c_str();

            if (trackerInfo->nvTracker_flag == NVTRACKER_CONFIG) {
                try {
                    trackerInfo->config_path = sub_root_val.get("config_path", "").asString();
                } catch (const std::exception &e) {
                    // Error handling: other exceptions
                    trackerInfo->nvTracker_log =
                        "NVTRACKER_CONFIG_UPDATE_FAIL, error: " + std::string(e.what());
                    trackerInfo->status = NVTRACKER_CONFIG_UPDATE_FAIL;
                    trackerInfo->err_info.code = StatusBadRequest;
                    return false;
                }
            }
        }
    } else {
        g_print("Unsupported REST API version\n");
    }
    return true;
}
