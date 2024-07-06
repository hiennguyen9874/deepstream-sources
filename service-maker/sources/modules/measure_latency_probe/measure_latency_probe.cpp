#include "measure_latency_probe.hpp"

#include "common_factory.hpp"
#include "custom_factory.hpp"
#include "plugin.h"

using namespace deepstream;

#define FACTORY_NAME "measure_latency_probe"

DS_CUSTOM_PLUGIN_DEFINE(measure_latency_probe,
                        "Custom probe to add measure Latency",
                        "0.1",
                        "Proprietary")

DS_CUSTOM_FACTORY_DEFINE(FACTORY_NAME,
                         "Latency measurement calculating custom probe factory",
                         "probe",
                         "this is a latency measurement custom probe factory",
                         "NVIDIA",
                         BufferProbe,
                         NvDsMeasureLatency)
