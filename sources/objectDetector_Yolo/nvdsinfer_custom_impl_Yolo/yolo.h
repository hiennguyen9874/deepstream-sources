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

#ifndef _YOLO_H_
#define _YOLO_H_

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "nvdsinfer_custom_impl.h"
#include "trt_utils.h"

/**
 * Holds all the file paths required to build a network.
 */
struct NetworkInfo {
    std::string networkType;
    std::string configFilePath;
    std::string wtsFilePath;
    std::string deviceType;
    std::string inputBlobName;
};

/**
 * Holds information about an output tensor of the yolo network.
 */
struct TensorInfo {
    std::string blobName;
    uint stride{0};
    uint gridSize{0};
    uint numClasses{0};
    uint numBBoxes{0};
    uint64_t volume{0};
    std::vector<uint> masks;
    std::vector<float> anchors;
    int bindingIndex{-1};
    float *hostBuffer{nullptr};
};

class Yolo : public IModelParser {
public:
    Yolo(const NetworkInfo &networkInfo);
    ~Yolo() override;
    bool hasFullDimsSupported() const override { return false; }
    const char *getModelName() const override
    {
        return m_ConfigFilePath.empty() ? m_NetworkType.c_str() : m_ConfigFilePath.c_str();
    }
    NvDsInferStatus parseModel(nvinfer1::INetworkDefinition &network) override;

    nvinfer1::ICudaEngine *createEngine(nvinfer1::IBuilder *builder,
                                        nvinfer1::IBuilderConfig *config);

protected:
    const std::string m_NetworkType;
    const std::string m_ConfigFilePath;
    const std::string m_WtsFilePath;
    const std::string m_DeviceType;
    const std::string m_InputBlobName;
    std::vector<TensorInfo> m_OutputTensors;
    std::vector<std::map<std::string, std::string>> m_ConfigBlocks;
    uint m_InputH;
    uint m_InputW;
    uint m_InputC;
    uint64_t m_InputSize;

    // TRT specific members
    std::vector<nvinfer1::Weights> m_TrtWeights;

private:
    NvDsInferStatus buildYoloNetwork(std::vector<float> &weights,
                                     nvinfer1::INetworkDefinition &network);
    std::vector<std::map<std::string, std::string>> parseConfigFile(const std::string cfgFilePath);
    void parseConfigBlocks();
    void destroyNetworkUtils();
};

#endif // _YOLO_H_
