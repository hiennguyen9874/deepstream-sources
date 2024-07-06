#include "sample_video_probe.hpp"

#include "common_factory.hpp"
#include "custom_factory.hpp"
#include "plugin.h"

using namespace deepstream;

DS_CUSTOM_FACTORY_DEFINE_PARAMS_BEGIN(probe_param_spec)
DS_CUSTOM_FACTORY_DEFINE_PARAM(font - size,
                               integer,
                               "font-size",
                               "size of the font to show the counter",
                               12)
DS_CUSTOM_FACTORY_DEFINE_PARAMS_END

#define FACTORY_NAME "sample_video_probe"

DS_CUSTOM_PLUGIN_DEFINE(sample_video_probe,
                        "this is a sample video buffer probe plugin",
                        "0.1",
                        "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_PARAMS(
    FACTORY_NAME,
    "sample video buffer probe factory",
    "probe",
    "this is a sample video buffer probe factory to create a object count marker",
    "NVIDIA",
    "",
    probe_param_spec,
    BufferProbe,
    CountMarker)