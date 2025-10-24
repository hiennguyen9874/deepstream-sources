#include <iostream>
#include <string>

#include "common_factory.hpp"
#include "pipeline.hpp"

#define KAFKA_PROTO_LIB_PATH "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
#define KAFKA_CONFIG_FILE \
    "/opt/nvidia/deepstream/deepstream/sources/libs/kafka_protocol_adaptor/cfg_kafka.txt"
#define MSGCONV_CONFIG_FILE                                                                \
    "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test5/configs/" \
    "dstest5_msgconv_sample_config.txt"
#define KAFKA_CONN_STR "localhost:9092"
#define KAFKA_TOPIC_LIST "sr-test"

#define CONFIG_FILE_PATH \
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml"

#define BATCHED_PUSH_TIMEOUT 33000
#define MUXER_WIDTH 1920
#define MUXER_HEIGHT 1080
#define TILER_WIDTH 1280
#define TILER_HEIGHT 720

using namespace deepstream;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <uri1> [uri2] ... [uriN]" << std::endl;
        return 0;
    }

    uint i, num_sources = argc - 1;
    std::string sink = "nveglglessink";

#if defined(__aarch64__)
    sink = "nv3dsink";
#endif

    try {
        Pipeline pipeline("deepstream-sr-test");

        auto object =
            CommonFactory::getInstance().createObject("smart_recording_action", "sr_action");
        auto *sr_action = dynamic_cast<SignalEmitter *>(object.get());
        if (!sr_action) {
            std::cerr << "Failed to create signal emitter" << std::endl;
            return -1;
        }

        sr_action->set("proto-lib", KAFKA_PROTO_LIB_PATH, "conn-str", KAFKA_CONN_STR,
                       "msgconv-config-file", MSGCONV_CONFIG_FILE, "proto-config-file",
                       KAFKA_CONFIG_FILE, "topic-list", KAFKA_TOPIC_LIST);

        for (i = 0; i < num_sources; i++) {
            std::string src_name = "src_";
            src_name += std::to_string(i);
            pipeline.add("nvurisrcbin", src_name, "uri", argv[i + 1], "smart-record", 1,
                         "smart-rec-cache", 20, "smart-rec-container", 0, "smart-rec-dir-path", ".",
                         "smart-rec-mode", 0);
            sr_action->attach("start-sr", pipeline[src_name]);
            sr_action->attach("stop-sr", pipeline[src_name]);
            pipeline[src_name].connectSignal("smart_recording_signal", "sr", "sr-done");
        }

        pipeline
            .add("nvstreammux", "mux", "batch-size", num_sources, "batched-push-timeout",
                 BATCHED_PUSH_TIMEOUT, "width", MUXER_WIDTH, "height", MUXER_HEIGHT)
            .add("nvinfer", "infer", "config-file-path", CONFIG_FILE_PATH, "batch-size",
                 num_sources)
            .add("nvmultistreamtiler", "tiler", "width", TILER_WIDTH, "height", TILER_HEIGHT)
            .add("nvvideoconvert", "converter")
            .add("nvdsosd", "osd")
            .add(sink, "sink", "sync", false)
            .link("mux", "infer", "tiler", "converter", "osd", "sink");

        for (i = 0; i < num_sources; i++) {
            std::string src = "src_" + std::to_string(i);
            pipeline.link({src, "mux"}, {"", "sink_%u"});
        }

        pipeline.start().wait();

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
