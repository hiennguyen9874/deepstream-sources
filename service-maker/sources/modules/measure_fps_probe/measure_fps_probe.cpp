#include "measure_fps_probe.hpp"

#include "common_factory.hpp"
#include "custom_factory.hpp"
#include "plugin.h"

using namespace deepstream;

#define FACTORY_NAME "measure_fps_probe"

DS_CUSTOM_PLUGIN_DEFINE(measure_fps_probe, "Custom probe to add measure FPS", "0.1", "Proprietary")

DS_CUSTOM_FACTORY_DEFINE(FACTORY_NAME,
                         "fps measurement calculating custom probe factory",
                         "probe",
                         "this is a fps measurement custom probe factory",
                         "NVIDIA",
                         BufferProbe,
                         FPSCounter)
