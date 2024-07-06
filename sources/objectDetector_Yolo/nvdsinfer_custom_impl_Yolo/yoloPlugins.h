#ifndef __YOLO_PLUGINS__
#define __YOLO_PLUGINS__

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>

#include "NvInferPlugin.h"

#define CHECK(status)                                                                              \
    {                                                                                              \
        if (status != 0) {                                                                         \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                     \
            abort();                                                                               \
        }                                                                                          \
    }

namespace {
const char *YOLOV3LAYER_PLUGIN_VERSION{"1"};
const char *YOLOV3LAYER_PLUGIN_NAME{"YoloLayerV3_TRT"};
} // namespace

class YoloLayerV3 : public nvinfer1::IPluginV2 {
public:
    YoloLayerV3(const void *data, size_t length);
    YoloLayerV3(const uint &numBoxes, const uint &numClasses, const uint &gridSize);
    const char *getPluginType() const noexcept override { return YOLOV3LAYER_PLUGIN_NAME; }
    const char *getPluginVersion() const noexcept override { return YOLOV3LAYER_PLUGIN_VERSION; }
    int getNbOutputs() const noexcept override { return 1; }

    nvinfer1::Dims getOutputDimensions(int index,
                                       const nvinfer1::Dims *inputs,
                                       int nbInputDims) noexcept override;

    bool supportsFormat(nvinfer1::DataType type,
                        nvinfer1::PluginFormat format) const noexcept override;

    void configureWithFormat(const nvinfer1::Dims *inputDims,
                             int nbInputs,
                             const nvinfer1::Dims *outputDims,
                             int nbOutputs,
                             nvinfer1::DataType type,
                             nvinfer1::PluginFormat format,
                             int maxBatchSize) noexcept override;

    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }
    int32_t enqueue(int32_t batchSize,
                    void const *const *inputs,
                    void *const *outputs,
                    void *workspace,
                    cudaStream_t stream) noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void *buffer) const noexcept override;
    void destroy() noexcept override { delete this; }
    nvinfer1::IPluginV2 *clone() const noexcept override;

    void setPluginNamespace(const char *pluginNamespace) noexcept override
    {
        m_Namespace = pluginNamespace;
    }
    virtual const char *getPluginNamespace() const noexcept override { return m_Namespace.c_str(); }

private:
    uint m_NumBoxes{0};
    uint m_NumClasses{0};
    uint m_GridSize{0};
    uint64_t m_OutputSize{0};
    std::string m_Namespace{""};
};

class YoloLayerV3PluginCreator : public nvinfer1::IPluginCreator {
public:
    YoloLayerV3PluginCreator() {}
    ~YoloLayerV3PluginCreator() {}

    const char *getPluginName() const noexcept override { return YOLOV3LAYER_PLUGIN_NAME; }
    const char *getPluginVersion() const noexcept override { return YOLOV3LAYER_PLUGIN_VERSION; }

    const nvinfer1::PluginFieldCollection *getFieldNames() noexcept override
    {
        std::cerr << "YoloLayerV3PluginCreator::getFieldNames is not implemented" << std::endl;
        return nullptr;
    }

    nvinfer1::IPluginV2 *createPlugin(const char *name,
                                      const nvinfer1::PluginFieldCollection *fc) noexcept override
    {
        std::cerr << "YoloLayerV3PluginCreator::getFieldNames is not implemented.\n";
        return nullptr;
    }

    nvinfer1::IPluginV2 *deserializePlugin(const char *name,
                                           const void *serialData,
                                           size_t serialLength) noexcept override
    {
        std::cout << "Deserialize yoloLayerV3 plugin: " << name << std::endl;
        return new YoloLayerV3(serialData, serialLength);
    }

    void setPluginNamespace(const char *libNamespace) noexcept override
    {
        m_Namespace = libNamespace;
    }
    const char *getPluginNamespace() const noexcept override { return m_Namespace.c_str(); }

private:
    std::string m_Namespace{""};
};

#endif // __YOLO_PLUGINS__
