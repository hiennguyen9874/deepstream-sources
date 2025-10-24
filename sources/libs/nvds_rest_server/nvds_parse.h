#ifndef _NVDS_PARSE_H_
#define _NVDS_PARSE_H_

#include <json/json.h>

bool nvds_rest_roi_parse(const Json::Value &in, NvDsServerRoiInfo *roi_info);
bool nvds_rest_dec_parse(const Json::Value &in, NvDsServerDecInfo *dec_info);
bool nvds_rest_enc_parse(const Json::Value &in, NvDsServerEncInfo *enc_info);
bool nvds_rest_conv_parse(const Json::Value &in, NvDsServerConvInfo *conv_info);
bool nvds_rest_mux_parse(const Json::Value &in, NvDsServerMuxInfo *mux_info);
bool nvds_rest_inferserver_parse(const Json::Value &in,
                                 NvDsServerInferServerInfo *inferserver_info);
bool nvds_rest_stream_parse(const Json::Value &in, NvDsServerStreamInfo *stream_info);
bool nvds_rest_infer_parse(const Json::Value &in, NvDsServerInferInfo *infer_info);
bool nvds_rest_nvtracker_parse(const Json::Value &in, NvDsServerNvTrackerInfo *trackerInfo);
bool nvds_rest_osd_parse(const Json::Value &in, NvDsServerOsdInfo *osd_info);
bool nvds_rest_analytics_parse(const Json::Value &in, NvDsServerAnalyticsInfo *analytics_info);
bool nvds_rest_app_instance_parse(const Json::Value &in,
                                  NvDsServerAppInstanceInfo *appinstance_info);

#endif
