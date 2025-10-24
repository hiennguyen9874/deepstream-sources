#include "nvds_parse.h"
#include "nvds_rest_server.h"

#define EMPTY_STRING ""

bool nvds_rest_inferserver_parse(const Json::Value &in, NvDsServerInferServerInfo *inferserver_info)
{
    if (inferserver_info->uri.find("/api/v1/") != std::string::npos) {
        for (Json::ValueConstIterator it = in.begin(); it != in.end(); ++it) {
            std::string root_val = it.key().asString().c_str();
            inferserver_info->root_key = root_val;

            const Json::Value sub_root_val = in[root_val]; // object values of root_key

            inferserver_info->stream_id =
                sub_root_val.get("stream_id", EMPTY_STRING).asString().c_str();
            if (inferserver_info->inferserver_flag == INFERSERVER_INTERVAL) {
                try {
                    inferserver_info->interval = sub_root_val.get("interval", 0).asUInt();
                    if (inferserver_info->interval > INT_MAX) {
                        inferserver_info->inferserver_log =
                            "INFERSERVER_INTERVAL_UPDATE_FAIL, interval value not parsed "
                            "correctly, Unsigned Integer. Range: 0 - 2147483647 ";
                        inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
                        inferserver_info->err_info.code = StatusBadRequest;
                        return false;
                    }
                } catch (const std::exception &e) {
                    // Error handling: other exceptions
                    inferserver_info->inferserver_log =
                        "INFERSERVER_INTERVAL_UPDATE_FAIL, error: " + std::string(e.what());
                    inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
                    inferserver_info->err_info.code = StatusBadRequest;
                    return false;
                }
            }
        }
    } else {
        g_print("Unsupported REST API version\n");
    }

    return true;
}
