/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cassert>
#include <cstring>
#include <iostream>

#include "nvdsinfer_custom_impl.h"

/* This is a custom parsing function for the TAO PeopleSemSegNet model
 * provided at NGC. */

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomPeopleSemSegNet(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float segmentationThreshold,
    unsigned int numClasses,
    int *classificationMap,
    float *&classProbabilityMap);

extern "C" bool NvDsInferParseCustomPeopleSemSegNet(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float segmentationThreshold,
    unsigned int numClasses,
    int *classificationMap,
    float *&classProbabilityMap)
{
    assert(classificationMap);

    auto layerFinder = [&outputLayersInfo](const std::string &name) -> const NvDsInferLayerInfo * {
        for (auto &layer : outputLayersInfo) {
            if (layer.dataType == INT32 && (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *classMapLayer = layerFinder("argmax_1"); // height x width x 1

    if (!classMapLayer) {
        std::cerr << "ERROR: Output layer argmax_1 not found in output tensors"
                  << " or was not of type INT32" << std::endl;
        return false;
    }

    NvDsInferDims outputDims = classMapLayer->inferDims;

    if (outputDims.numDims != 3U) {
        std::cerr << "Network output number of dims is : " << outputDims.numDims
                  << " expected is 3." << std::endl;
        return false;
    }

    if (outputDims.d[0] != networkInfo.height || outputDims.d[1] != networkInfo.width ||
        outputDims.d[2] != 1) {
        std::cerr << "Incorrect output layer dimensions : " << outputDims.d[0] << " expected is "
                  << networkInfo.height << "x" << networkInfo.width << "x1." << std::endl;
        return false;
    }

    classProbabilityMap = nullptr;
    memcpy(classificationMap, classMapLayer->buffer,
           sizeof(int32_t) * networkInfo.width * networkInfo.height);

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_SEM_SEGMENTATION_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomPeopleSemSegNet);
