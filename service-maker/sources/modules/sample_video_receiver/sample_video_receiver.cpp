#include "sample_video_receiver.hpp"

#include "common_factory.hpp"
#include "custom_factory.hpp"
#include "plugin.h"

using namespace deepstream;

#define FACTORY_NAME "sample_video_receiver"

DS_CUSTOM_PLUGIN_DEFINE(sample_video_receiver,
                        "this is a sample data receiver plugin",
                        "0.1",
                        "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_SIGNALS(
    FACTORY_NAME,
    "sample video data receiver factory",
    "signal",
    "this is a sample video data receiver factory to create a data receiver to count objects",
    "NVIDIA",
    "new-sample",
    DataReceiver,
    ObjectCounter)