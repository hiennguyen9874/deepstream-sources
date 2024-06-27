/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <algorithm>

#include "nvdsinfer_context.h"
#include "nvdsinfer_custom_impl.h"
#include "yolo.h"
#include "yoloPlugins.h"

#define USE_CUDA_ENGINE_GET_API 1

static bool getYoloNetworkInfo(NetworkInfo &networkInfo,
                               const NvDsInferContextInitParams *initParams)
{
    std::string yoloCfg = initParams->customNetworkConfigFilePath;
    std::string yoloType;

    std::transform(yoloCfg.begin(), yoloCfg.end(), yoloCfg.begin(),
                   [](uint8_t c) { return std::tolower(c); });

    if (yoloCfg.find("yolov2") != std::string::npos) {
        if (yoloCfg.find("yolov2-tiny") != std::string::npos)
            yoloType = "yolov2-tiny";
        else
            yoloType = "yolov2";
    } else if (yoloCfg.find("yolov3") != std::string::npos) {
        if (yoloCfg.find("yolov3-tiny") != std::string::npos)
            yoloType = "yolov3-tiny";
        else
            yoloType = "yolov3";
    } else {
        std::cerr << "Yolo type is not defined from config file name:" << yoloCfg << std::endl;
        return false;
    }

    networkInfo.networkType = yoloType;
    networkInfo.configFilePath = initParams->customNetworkConfigFilePath;
    networkInfo.wtsFilePath = initParams->modelFilePath;
    networkInfo.deviceType = (initParams->useDLA ? "kDLA" : "kGPU");
    networkInfo.inputBlobName = "data";

    if (networkInfo.configFilePath.empty() || networkInfo.wtsFilePath.empty()) {
        std::cerr << "Yolo config file or weights file is NOT specified." << std::endl;
        return false;
    }

    if (!fileExists(networkInfo.configFilePath) || !fileExists(networkInfo.wtsFilePath)) {
        std::cerr << "Yolo config file or weights file is NOT exist." << std::endl;
        return false;
    }

    return true;
}

#if !USE_CUDA_ENGINE_GET_API
IModelParser *NvDsInferCreateModelParser(const NvDsInferContextInitParams *initParams)
{
    NetworkInfo networkInfo;
    if (!getYoloNetworkInfo(networkInfo, initParams)) {
        return nullptr;
    }

    return new Yolo(networkInfo);
}
#else
extern "C" bool NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder *const builder,
                                           nvinfer1::IBuilderConfig *const builderConfig,
                                           const NvDsInferContextInitParams *const initParams,
                                           nvinfer1::DataType dataType,
                                           nvinfer1::ICudaEngine *&cudaEngine);

extern "C" bool NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder *const builder,
                                           nvinfer1::IBuilderConfig *const builderConfig,
                                           const NvDsInferContextInitParams *const initParams,
                                           nvinfer1::DataType dataType,
                                           nvinfer1::ICudaEngine *&cudaEngine)
{
    NetworkInfo networkInfo;
    if (!getYoloNetworkInfo(networkInfo, initParams)) {
        return false;
    }

    Yolo yolo(networkInfo);
    cudaEngine = yolo.createEngine(builder, builderConfig);
    if (cudaEngine == nullptr) {
        std::cerr << "Failed to build cuda engine on " << networkInfo.configFilePath << std::endl;
        return false;
    }

    return true;
}
#endif
