#include <iostream>
#include <string>

#include "pipeline.hpp"

#define MUXER_WIDTH 1280
#define MUXER_HEIGHT 720
#define CONFIG_FILE_PATH                                                           \
    "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test4/" \
    "dstest4_pgie_config.yml"
#define CONFIG_MSGCONV_PATH                                                        \
    "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test4/" \
    "dstest4_msgconv_config.yml"
#define KAFKA_PROTO_LIB "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
#define MSGBROKER_CONN_STR "localhost;9092"
#define MGSBROKER_TOPIC "test4app"

using namespace deepstream;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <YAML config file>" << std::endl;
        std::cout << "OR: " << argv[0] << " <H264 filename>" << std::endl;
        return 0;
    }

    std::string sink = "nveglglessink";

#if defined(__aarch64__)
    sink = "nv3dsink";
#endif

    try {
        std::string file = argv[1];
        std::string suffix = "yaml";
        if (std::equal(suffix.rbegin(), suffix.rend(), file.rbegin())) {
            Pipeline pipeline("deepstream-test4", file);
            pipeline.start().wait();
        } else {
            Pipeline pipeline("deepstream-test4");
            pipeline.add("filesrc", "src", "location", argv[1])
                .add("h264parse", "parser")
                .add("nvv4l2decoder", "decoder")
                .add("nvstreammux", "mux", "batch-size", 1, "width", MUXER_WIDTH, "height",
                     MUXER_HEIGHT)
                .add("nvinfer", "infer", "config-file-path", CONFIG_FILE_PATH)
                .add("nvvideoconvert", "converter")
                .add("nvdsosd", "osd")
                .add("tee", "tee")
                .add("queue", "queue1")
                .add("queue", "queue2")
                .add("nvmsgconv", "msgconv", "config", CONFIG_MSGCONV_PATH)
                .add("nvmsgbroker", "msgbroker", "conn-str", MSGBROKER_CONN_STR, "proto-lib",
                     KAFKA_PROTO_LIB, "sync", false, "topic", MGSBROKER_TOPIC)
                .add(sink, "sink")
                .link("src", "parser", "decoder")
                .link({"decoder", "mux"}, {"", "sink_%u"})
                .link("mux", "infer", "converter", "osd", "tee")
                .link("tee", "queue1", "sink")
                .link("tee", "queue2", "msgconv", "msgbroker")
                .attach("osd", "add_message_meta_probe", "metadata generator")
                .start()
                .wait();
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
