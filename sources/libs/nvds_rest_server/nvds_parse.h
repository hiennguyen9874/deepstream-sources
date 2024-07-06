#ifndef _NVDS_PARSE_H_
#define _NVDS_PARSE_H_

#include <jsoncpp/json/json.h>

bool nvds_rest_roi_parse(const Json::Value &in, NvDsRoiInfo *roi_info);
bool nvds_rest_dec_parse(const Json::Value &in, NvDsDecInfo *dec_info);
bool nvds_rest_stream_parse(const Json::Value &in, NvDsStreamInfo *stream_info);
bool nvds_rest_infer_parse(const Json::Value &in, NvDsInferInfo *infer_info);

#endif
