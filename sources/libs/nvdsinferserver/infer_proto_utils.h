/**
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#ifndef __NVDSINFER_SERVER_PROTO_UTILS_H__
#define __NVDSINFER_SERVER_PROTO_UTILS_H__

#include <dlfcn.h>
#include <infer_datatypes.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#include <cassert>
#include <condition_variable>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#include "nvbufsurftransform.h"

#pragma GCC diagnostic push
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wrestrict"
#endif
#include "nvdsinferserver_config.pb.h"
#include "nvdsinferserver_plugin.pb.h"
#pragma GCC diagnostic pop

namespace ic = nvdsinferserver::config;

namespace nvdsinferserver {

InferDataType dataTypeFromDsProto(ic::TensorDataType dt);

InferTensorOrder tensorOrderFromDsProto(ic::TensorOrder o);
InferMediaFormat mediaFormatFromDsProto(ic::MediaFormat f);
InferMemType memTypeFromDsProto(ic::MemoryType t);

NvBufSurfTransform_Compute computeHWFromDsProto(ic::FrameScalingHW h);
NvBufSurfTransform_Inter scalingFilterFromDsProto(uint32_t filter);

bool validateProtoConfig(ic::InferenceConfig &c, const std::string &path);
bool compareModelRepo(const ic::TritonModelRepo &repoA, const ic::TritonModelRepo &repoB);

inline bool hasTriton(const ic::BackendParams &params)
{
    return params.has_triton() || params.has_trt_is();
}
inline const ic::TritonParams &getTritonParam(const ic::BackendParams &params)
{
    if (params.has_triton()) {
        return params.triton();
    } else {
        assert(params.has_trt_is());
        return params.trt_is();
    }
}
inline ic::TritonParams *mutableTriton(ic::BackendParams &params)
{
    if (params.has_triton()) {
        return params.mutable_triton();
    } else {
        assert(params.has_trt_is());
        return params.mutable_trt_is();
    }
}

} // namespace nvdsinferserver

#endif
