#ifndef _NVDS_PARSE_H_
#define _NVDS_PARSE_H_

#include <json/json.h>

bool nvds_rest_roi_parse(const Json::Value &in, NvDsRoiInfo *roi_info);
bool nvds_rest_dec_parse(const Json::Value &in, NvDsDecInfo *dec_info);
bool nvds_rest_enc_parse(const Json::Value &in, NvDsEncInfo *enc_info);
bool nvds_rest_conv_parse(const Json::Value &in, NvDsConvInfo *conv_info);
bool nvds_rest_mux_parse(const Json::Value &in, NvDsMuxInfo *mux_info);
bool nvds_rest_inferserver_parse(const Json::Value &in, NvDsInferServerInfo *inferserver_info);
bool nvds_rest_stream_parse(const Json::Value &in, NvDsStreamInfo *stream_info);
bool nvds_rest_infer_parse(const Json::Value &in, NvDsInferInfo *infer_info);
bool nvds_rest_osd_parse(const Json::Value &in, NvDsOsdInfo *osd_info);
bool nvds_rest_app_instance_parse(const Json::Value &in, NvDsAppInstanceInfo *appinstance_info);

#endif
