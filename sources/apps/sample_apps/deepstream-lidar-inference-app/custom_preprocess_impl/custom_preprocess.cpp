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

#include <cuda_runtime_api.h>

// inlcude all ds3d hpp header files
#include <ds3d/common/common.h>

#include <ds3d/common/hpp/datamap.hpp>
#include <ds3d/common/hpp/frame.hpp>
#include <ds3d/common/hpp/lidar_custom_process.hpp>
#include <ds3d/common/hpp/yaml_config.hpp>
#include <string>

#include "infer_datatypes.h"
#include "nvdsinfer.h"
using namespace ds3d;

using namespace nvdsinferserver;

extern "C" IInferCustomPreprocessor *CreateInferServerCustomPreprocess();

#ifndef INFER_ASSERT
#define INFER_ASSERT(expr)                                                     \
    do {                                                                       \
        if (!(expr)) {                                                         \
            fprintf(stderr, "%s:%d ASSERT(%s) \n", __FILE__, __LINE__, #expr); \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
#endif

#define checkCudaErrors(status)                                                                    \
    {                                                                                              \
        if (status != 0) {                                                                         \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " at line " << __LINE__ \
                      << " in file " << __FILE__ << " error status: " << status << std::endl;      \
            abort();                                                                               \
        }                                                                                          \
    }

class NvInferServerCustomPreProcess : public IInferCustomPreprocessor {
public:
    ~NvInferServerCustomPreProcess() final = default;
    NvDsInferStatus preproc(GuardDataMap &dataMap, SharedIBatchArray batchArray) override
    {
        FrameGuard lidarFrame;
        const IOptions *inOptions = batchArray->getOptions();
        std::string key;
        if (inOptions && inOptions->hasValue(kLidarXYZI)) {
            INFER_ASSERT(inOptions->getValue<std::string>(kLidarXYZI, key) == NVDSINFER_SUCCESS);
        }
        ErrCode code = dataMap.getGuardData(key, lidarFrame);
        if (!isGood(code)) {
            std::cout << "dataMap getGuardData kLidarFrame failed" << std::endl;
            return NVDSINFER_TENSORRT_ERROR;
        }

        if (!_configParsed && inOptions->hasValue(kLidarInferenceParas)) {
            std::string inferConfigRaw;
            NvDsInferStatus ret =
                inOptions->getValue<std::string>(kLidarInferenceParas, inferConfigRaw);
            if (ret != NVDSINFER_SUCCESS) {
                std::cerr << "preprocess query key: " << kLidarInferenceParas << " failed.\n";
            }
            preprocessConfigParse(inferConfigRaw);
            _configParsed = true;
        }

        INFER_ASSERT(batchArray->getSize() > 1);
        const IBatchBuffer *buf = batchArray->getBuffer(0);
        //[0-255] to [0-1]
        InferBufferDescription des = buf->getBufDesc();
        int numPoints = std::accumulate(des.dims.d, des.dims.d + des.dims.numDims - 1, 1,
                                        [](int s, int i) { return s * i; });
        int elementSize = des.dims.d[des.dims.numDims - 1];
        INFER_ASSERT(elementSize == 4);

        float *frame = (float *)lidarFrame->base();
        // normalize intensity values
        for (int j = 0; j < numPoints; j++) {
            float &val = frame[j * elementSize + 3];
            val = (val - _offsets) * _scaleFactor;
        }
        // copy preprocess data to GpuCuda or CpuCuda
        checkCudaErrors(cudaMemcpy(buf->getBufPtr(0), lidarFrame->base(), lidarFrame->bytes(),
                                   cudaMemcpyDefault));

        // add the second input.
        buf = batchArray->getBuffer(1);
        unsigned int points_size = numPoints;
        checkCudaErrors(
            cudaMemcpy(buf->getBufPtr(0), &points_size, sizeof(unsigned int), cudaMemcpyDefault));
        return NVDSINFER_SUCCESS;
    }

    void preprocessConfigParse(const std::string &configRaw)
    {
        YAML::Node node = YAML::Load(configRaw);
        if (node["preprocess"]) {
            auto preproc = node["preprocess"];
            if (preproc["scale_factor"]) {
                _scaleFactor = preproc["scale_factor"].as<float>();
            }
            if (preproc["offsets"]) {
                _offsets = preproc["offsets"].as<float>();
            }
        }
    }

private:
    bool _configParsed = false;
    float _scaleFactor = 1.0f;
    float _offsets = 0.0f;
};

extern "C" {
IInferCustomPreprocessor *CreateInferServerCustomPreprocess()
{
    return new NvInferServerCustomPreProcess();
}
}
