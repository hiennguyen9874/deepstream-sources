#include "add_message_meta_probe.hpp"

#include "common_factory.hpp"
#include "custom_factory.hpp"
#include "plugin.h"

using namespace std;
using namespace deepstream;

#define FACTORY_NAME "add_message_meta_probe"

DS_CUSTOM_PLUGIN_DEFINE(add_message_meta_probe,
                        "Custom probe to add NVDS_META_EVENT_MSG data to buffer",
                        "0.1",
                        "Proprietary")

DS_CUSTOM_FACTORY_DEFINE(FACTORY_NAME,
                         "message meta adding custom probe factory",
                         "probe",
                         "this is a message meta adding custom probe factory",
                         "NVIDIA",
                         BufferProbe,
                         MsgMetaGenerator)
