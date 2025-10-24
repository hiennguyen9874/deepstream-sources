#include "kitti_dump_probe.hpp"

#include "common_factory.hpp"
#include "custom_factory.hpp"
#include "plugin.h"

using namespace std;
using namespace deepstream;

#define FACTORY_NAME "kitti_dump_probe"

DS_CUSTOM_FACTORY_DEFINE_PARAMS_BEGIN(probe_param_spec)
DS_CUSTOM_FACTORY_DEFINE_PARAM(kitti - dir,
                               string,
                               "kitti-dir",
                               "directory of kitti output",
                               "/tmp/kitti")
DS_CUSTOM_FACTORY_DEFINE_PARAM(tracker - kitti - output,
                               boolean,
                               "tracker-kitti-output",
                               "enable tracker kitti output",
                               false)
DS_CUSTOM_FACTORY_DEFINE_PARAMS_END

DS_CUSTOM_PLUGIN_DEFINE(kitti_dump_probe, "Custom probe for kitti dump", "0.1", "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_PARAMS(FACTORY_NAME,
                                     "kitti dump adding custom probe factory",
                                     "probe",
                                     "this is a kitti dumping custom probe factory",
                                     "NVIDIA",
                                     "",
                                     probe_param_spec,
                                     BufferProbe,
                                     NvDsKittiDump)