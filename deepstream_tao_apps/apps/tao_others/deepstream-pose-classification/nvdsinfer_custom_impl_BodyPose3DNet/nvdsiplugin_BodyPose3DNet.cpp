#include "factoryBodyPose3DNet.h"
#include "nvdsinfer_custom_impl.h"

bool NvDsInferPluginFactoryCaffeGet(NvDsInferPluginFactoryCaffe &pluginFactory,
                                    NvDsInferPluginFactoryType &type)
{
    type = PLUGIN_FACTORY_V2;
    pluginFactory.pluginFactoryV2 = new FRCNNPluginFactory;

    return true;
}

void NvDsInferPluginFactoryCaffeDestroy(NvDsInferPluginFactoryCaffe &pluginFactory)
{
    FRCNNPluginFactory *factory = static_cast<FRCNNPluginFactory *>(pluginFactory.pluginFactoryV2);
    factory->destroyPlugin();
    delete factory;
}
