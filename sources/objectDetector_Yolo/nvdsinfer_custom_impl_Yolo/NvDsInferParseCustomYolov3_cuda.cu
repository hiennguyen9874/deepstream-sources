/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "nvdsinfer_custom_impl.h"
#include "trt_utils.h"
// whr tag:how about the static value in function
static const int NUM_CLASSES_YOLO = 80;
std::vector<float> tempkANCHORS_v = {10.0, 13.0, 16.0,  30.0,  33.0, 23.0,  30.0,  61.0,  62.0,
                                     45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};
std::vector<int> tempkMASKS_v = {6, 7, 8, 3, 4, 5, 0, 1, 2};
thrust::device_vector<float> d_kANCHORS_v(tempkANCHORS_v);
thrust::device_vector<int> d_kMASKS_v(tempkMASKS_v);
thrust::device_vector<NvDsInferParseObjectInfo> objects_v(1083 + 4332 + 17328);

static inline std::vector<const NvDsInferLayerInfo *> SortLayers(
    const std::vector<NvDsInferLayerInfo> &outputLayersInfo)
{
    std::vector<const NvDsInferLayerInfo *> outLayers;
    for (auto const &layer : outputLayersInfo) {
        outLayers.push_back(&layer);
    }
    std::sort(outLayers.begin(), outLayers.end(),
              [](const NvDsInferLayerInfo *a, const NvDsInferLayerInfo *b) {
                  return a->inferDims.d[1] < b->inferDims.d[1];
              });
    return outLayers;
}

//*******************belows are cuda kernels*****************
__global__ void decodeYoloV3Tensor_cuda(NvDsInferParseObjectInfo *binfo,
                                        const float *detections,
                                        const int *mask,
                                        const float *anchors,
                                        const uint gridSizeW,
                                        const uint gridSizeH,
                                        const uint stride,
                                        const uint numBBoxes,
                                        const uint numOutputClasses,
                                        float netW,
                                        float netH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gridSizeW * gridSizeH * numBBoxes) {
        int b = idx % numBBoxes;
        int x = (idx / numBBoxes) % gridSizeW;
        int y = (idx / numBBoxes) / gridSizeW;
        const float pw = anchors[mask[b] * 2];
        const float ph = anchors[mask[b] * 2 + 1];
        const int numGridCells = gridSizeH * gridSizeW;
        const int bbindex = y * gridSizeW + x;
        const float bx = x + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 0)];
        const float by = y + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 1)];
        const float bw = pw * detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 2)];
        const float bh = ph * detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 3)];
        const float objectness =
            detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 4)];
        float maxProb = 0.0f;
        unsigned int maxIndex = (unsigned int)-1;
        for (uint i = 0; i < numOutputClasses; ++i) {
            float prob =
                (detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + (5 + i))]);
            if (prob > maxProb) {
                maxProb = prob;
                maxIndex = i;
            }
        }
        maxProb = objectness * maxProb;
        float xCenter = bx * stride;
        float yCenter = by * stride;
        float x0 = xCenter - bw / 2;
        float y0 = yCenter - bh / 2;
        float x1 = x0 + bw;
        float y1 = y0 + bh;
        // x0 = fmaxf(float(0.0), x0);
        x0 = fminf(float(netW), fmaxf(float(0.0), x0));
        y0 = fminf(float(netH), fmaxf(float(0.0), y0));
        x1 = fminf(float(netW), fmaxf(float(0.0), x1));
        y1 = fminf(float(netH), fmaxf(float(0.0), y1));
        binfo[idx].left = x0;
        binfo[idx].top = y0;
        binfo[idx].width = fminf(float(netW), fmaxf(float(0.0), x1 - x0));
        binfo[idx].height = fminf(float(netH), fmaxf(float(0.0), y1 - y0));
        binfo[idx].detectionConfidence = maxProb;
        binfo[idx].classId = maxIndex;
    }
}

bool NvDsInferParseYoloV3_cuda(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                               NvDsInferNetworkInfo const &networkInfo,
                               NvDsInferParseDetectionParams const &detectionParams,
                               std::vector<NvDsInferParseObjectInfo> &objectList,
                               const thrust::device_vector<float> &anchors,
                               const thrust::device_vector<int> &masks)
{
    const uint kNUM_BBOXES = 3;
    const std::vector<const NvDsInferLayerInfo *> sortedLayers = SortLayers(outputLayersInfo);
    if (sortedLayers.size() * 3 != masks.size()) {
        std::cerr << "ERROR: yoloV3 output layer.size: " << sortedLayers.size()
                  << " does not match mask.size: " << masks.size() << std::endl;
        return false;
    }
    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured) {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    uint layer1size = 0, layer2size = 0, layer3size = 0;

    for (uint idx = 0; idx < masks.size() / 3; ++idx) {
        const NvDsInferLayerInfo &layer = *sortedLayers[idx]; // 255 x Grid x Grid
        assert(layer.inferDims.numDims == 3);
        const uint gridSizeH = layer.inferDims.d[1];
        const uint gridSizeW = layer.inferDims.d[2];
        const uint stride = DIVUP(networkInfo.width, gridSizeW);
        assert(stride == DIVUP(networkInfo.height, gridSizeH));
        int offset = 0;
        if (idx == 0) {
            layer1size = gridSizeH * gridSizeW * layer.inferDims.numDims;
            offset = 0;
        } else if (idx == 1) {
            offset = layer1size;
            layer2size = gridSizeH * gridSizeW * layer.inferDims.numDims;
        } else if (idx == 2) {
            layer3size = gridSizeH * gridSizeW * layer.inferDims.numDims;
            offset = layer2size + layer1size;
        }
        int BLOCKSIZE = 1024;
        int GRIDSIZE = ((gridSizeH * gridSizeW * 3 - 1) / BLOCKSIZE) + 1;
        decodeYoloV3Tensor_cuda<<<GRIDSIZE, BLOCKSIZE>>>(
            thrust::raw_pointer_cast(objects_v.data()) + offset, (const float *)(layer.buffer),
            thrust::raw_pointer_cast(masks.data()) + idx * 3,
            thrust::raw_pointer_cast(anchors.data()), gridSizeW, gridSizeH, stride, kNUM_BBOXES,
            NUM_CLASSES_YOLO, static_cast<float>(networkInfo.width),
            static_cast<float>(networkInfo.height));
    }
    objectList.resize(layer1size + layer2size + layer3size);
    thrust::copy(objects_v.begin(), objects_v.end(), objectList.begin()); // the same as cudamemcpy

    return true;
}

extern "C" bool NvDsInferParseCustomYoloV3_cuda(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    bool ret = NvDsInferParseYoloV3_cuda(outputLayersInfo, networkInfo, detectionParams, objectList,
                                         d_kANCHORS_v, d_kMASKS_v);
    return ret;
}
