#include "ensemble_render.hpp"

#include "common_factory.hpp"
#include "custom_factory.hpp"
#include "plugin.h"

using namespace deepstream;

#define FACTORY_NAME "ensemble_render"

DS_CUSTOM_FACTORY_DEFINE_PARAMS_BEGIN(param_spec)
DS_CUSTOM_FACTORY_DEFINE_PARAM(config - path,
                               path,
                               "file path for data",
                               "the file contains the data for feeding the appsrc",
                               "")

DS_CUSTOM_FACTORY_DEFINE_PARAMS_END

DS_CUSTOM_PLUGIN_DEFINE(ensemble_render,
                        "this is a data receiver plugin for rendering",
                        "0.1",
                        "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_PARAMS(FACTORY_NAME,
                                     "sample data receiver factory",
                                     "signal",
                                     "this is a data receiver factory to render pipeline results",
                                     "NVIDIA",
                                     "new-sample",
                                     param_spec,
                                     DataReceiver,
                                     EnsembleRender)
