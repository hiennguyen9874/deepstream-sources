#include <string.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "common_factory.hpp"
#include "lib/model_engine_watch_otf_trigger.hpp"
#include "lib/perf_monitor.hpp"
#include "pipeline.hpp"
#include "source_config.hpp"
#include "tiler_event_handler.hpp"

using namespace deepstream;

class MsgMetaGenerator : public BufferProbe::IBatchMetadataOperator {
public:
    MsgMetaGenerator(const std::map<uint32_t, SensorInfo> &sensor_map,
                     const std::vector<std::string> labels,
                     int frame_interval = 1)
        : sensor_map_(sensor_map), labels_(labels), frame_interval_(frame_interval)
    {
    }
    virtual ~MsgMetaGenerator() {}

    virtual probeReturn handleData(BufferProbe &probe, BatchMetadata &data)
    {
        FrameMetadata::Iterator frame_itr;
        for (data.initiateIterator(frame_itr); !frame_itr->done(); frame_itr->next()) {
            ObjectMetadata::Iterator obj_itr;
            for ((*frame_itr)->initiateIterator(obj_itr); !obj_itr->done(); obj_itr->next()) {
                if (0 == (frames_ % frame_interval_)) {
                    EventMessageUserMetadata event_user_meta;
                    if (data.acquire(event_user_meta)) {
                        auto source_id = (*frame_itr)->sourceId();
                        auto itr = sensor_map_.find(source_id);
                        if (itr != sensor_map_.end()) {
                            const std::string sensor = itr->second.sensor_id;
                            const std::string uri = itr->second.uri;
                            event_user_meta.generate(**obj_itr, **frame_itr, sensor, uri, labels_);
                            (*frame_itr)->append(event_user_meta);
                        } else {
                            std::cout << "Warning: sensor with source id " << source_id
                                      << " doesn't exist" << std::endl;
                        }
                    }
                }
            }
            frames_++;
        }

        return probeReturn::Probe_Ok;
    }

protected:
    int frames_ = 0;
    const std::map<uint32_t, SensorInfo> &sensor_map_;
    std::vector<std::string> labels_;
    int frame_interval_;
};

static void usage()
{
    std::cout << "Usage: deepstream-test5-app -s source-lists -c config-files [-l label-file] "
                 "[--perf-measurement-interval-sec] [--hide-stream-name]"
              << std::endl;
}

static std::string parse_config(std::vector<std::string> &arguments, const std::string flag)
{
    std::string str;
    auto itr = std::find(arguments.begin(), arguments.end(), flag);
    if (itr != arguments.end()) {
        auto next_itr = itr + 1;
        if (next_itr != arguments.end()) {
            str = *next_itr;
        }
    }
    return str;
}

static std::vector<std::string> split(const std::string &input, char delimiter)
{
    std::vector<std::string> result;
    std::istringstream stream(input);
    std::string token;

    while (std::getline(stream, token, delimiter)) {
        result.push_back(token);
    }

    return result;
}

static std::vector<std::string> parse_labels(const std::string &filename)
{
    std::ifstream file(filename);   // Open the file
    std::vector<std::string> lines; // Vector to store lines

    if (file.is_open()) { // Check if the file is open
        std::string line;
        while (std::getline(file, line)) { // Read each line
            std::string result = line;
            std::transform(line.begin(), line.end(), result.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            lines.push_back(result); // Store the line in the vector
        }
        file.close(); // Close the file
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }

    return lines;
}

int main(int argc, char *argv[])
{
    if (argc < 5) {
        usage();
        return 0;
    }

    std::string model_engine_file;
    uint32_t perf_interval = 1;
    std::vector<std::string> arguments;
    for (int n = 1; n < argc; n++) {
        arguments.push_back(argv[n]);
    }
    // find the config files
    std::vector<std::string> config_files = split(parse_config(arguments, "-c"), ',');
    std::vector<std::string> source_config_files = split(parse_config(arguments, "-s"), ',');
    std::string label_file = parse_config(arguments, "-l");
    std::string perf_interval_sec = parse_config(arguments, "--perf-measurement-interval-sec");
    bool hide_stream_name =
        std::find(arguments.begin(), arguments.end(), "--hide-stream-name") != arguments.end();
    if (config_files.empty() || source_config_files.empty()) {
        usage();
        return 0;
    }
    if (config_files.size() != source_config_files.size()) {
        std::cout << "The number of pipeline configuration file must be equal to that of the "
                     "source list file"
                  << std::endl;
        return 0;
    }

    std::vector<std::map<uint32_t, SensorInfo>> sensor_maps(config_files.size());
    std::vector<std::string> labels;
    if (!label_file.empty()) {
        labels = parse_labels(label_file);
    }

    try {
        std::vector<std::unique_ptr<Pipeline>> pipelines;
        std::vector<std::unique_ptr<NvDsModelEngineWatchOTFTrigger>> otf_triggers;
        std::vector<std::unique_ptr<NvDsTilerEventHandler>> tiler_event_handlers;
        std::vector<std::unique_ptr<PerfMonitor>> perf_monitors;
        for (unsigned int idx = 0; idx < config_files.size(); idx++) {
            std::map<uint32_t, SensorInfo> &sensor_map = sensor_maps[idx];
            SourceConfig source_config(source_config_files[idx]);
            std::string pipeline_name = "deepstream-test5-";
            pipeline_name += std::to_string(idx);
            pipelines.push_back(
                std::make_unique<Pipeline>(pipeline_name.c_str(), config_files[idx]));
            Pipeline &pipeline = *pipelines.back().get();
            std::string kafka_proto_lib_path;
            std::string msgbroker_conn_str;
            std::string msgconv_config_path;
            std::string msgbroker_config_path;
            uint32_t tiler_width = 0;
            uint32_t tiler_height = 0;
            std::string pgie = "pgie";

            pipeline["msgbroker"].getProperty("config", msgbroker_config_path, "proto-lib",
                                              kafka_proto_lib_path, "conn-str", msgbroker_conn_str);
            pipeline["msgconv"].getProperty("config", msgconv_config_path);
            pipeline["tiler"].getProperty("width", tiler_width, "height", tiler_height);
            pipeline[pgie].getProperty("model-engine-file", model_engine_file);

            // create the smart recording action
            auto object =
                CommonFactory::getInstance().createObject("smart_recording_action", "sr_action");
            auto *sr_action = dynamic_cast<SignalEmitter *>(object.get());
            if (!sr_action) {
                std::cerr << "Failed to create signal emitter" << std::endl;
                return -1;
            }

            sr_action->set("proto-lib", kafka_proto_lib_path, "conn-str", msgbroker_conn_str,
                           "msgconv-config-file", msgconv_config_path, "proto-config-file",
                           msgbroker_config_path, "topic-list", "test5-sr");

            std::string checkpoint = "checkpoint";
            if (source_config.useMultiUriSrcBin()) {
                // nvmultiurisrcbin used
                pipeline.add("nvmultiurisrcbin", "multiurisrcbin");
                pipeline["multiurisrcbin"]
                    .set(source_config.getProperties())
                    .set("uri-list", source_config.listUris())
                    .set("sensor-id-list", source_config.listSensorIds())
                    .set("sensor-name-list", source_config.listSensorNames());

                pipeline.link("multiurisrcbin", pgie);
            } else {
                // nvurisrcbin used
                auto &properties = source_config.getProperties();
                pipeline.add("nvstreammux", "mux", "batch-size", source_config.nSources(), "width",
                             tiler_width, "height", tiler_height);
                if (properties["gpu-id"]) {
                    pipeline["mux"].set("gpu-id", properties["gpu-id"].as<int>());
                }
                // create and add all the required elements
                for (size_t i = 0; i < source_config.nSources(); i++) {
                    // multiple sources added
                    std::string src_name = "src_";
                    src_name += std::to_string(i);
                    std::string uri = source_config.getSensorInfo(i).uri;
                    pipeline.add("nvurisrcbin", src_name, "uri", uri.c_str())
                        .link({src_name, "mux"}, {"vsrc_%u", ""});
                    pipeline[src_name].set(properties);
                    sensor_map[i] = SensorInfo{uri, std::to_string(i), src_name};
                }
                pipeline.link("mux", pgie);
            }
            pipeline.attach(checkpoint, new BufferProbe("message_generator",
                                                        new MsgMetaGenerator(sensor_map, labels)));

            // create model update watcher
            otf_triggers.push_back(std::make_unique<NvDsModelEngineWatchOTFTrigger>(
                &pipeline[pgie], model_engine_file));
            NvDsModelEngineWatchOTFTrigger &otf_trigger = *otf_triggers.back().get();

            // create the event monitor for the tiler
            tiler_event_handlers.push_back(std::make_unique<NvDsTilerEventHandler>(
                &pipeline["tiler"], &pipeline["osd"], &pipeline["sink"]));
            NvDsTilerEventHandler &tiler_event_handler = *tiler_event_handlers.back().get();

            if (!source_config.useMultiUriSrcBin()) {
                // add the smart recording controll to all the sources
                for (size_t i = 0; i < source_config.nSources(); i++) {
                    std::string src_name = "src_";
                    src_name += std::to_string(i);
                    sr_action->attach("start-sr", pipeline[src_name]);
                    sr_action->attach("stop-sr", pipeline[src_name]);
                    pipeline[src_name].connectSignal("smart_recording_signal", "sr", "sr-done");
                }
            }

            // performance monitor
            if (!perf_interval_sec.empty()) {
                perf_interval = std::stoi(perf_interval_sec);
            }

            perf_monitors.push_back(std::make_unique<PerfMonitor>(
                source_config.nSources(), perf_interval,
                source_config.useMultiUriSrcBin() ? "nvmultiurisrcbin" : "nvurisrcbin",
                !hide_stream_name));
            PerfMonitor &perf_monitor = *perf_monitors.back().get();
            perf_monitor.apply(pipeline["tiler"], "sink");

            // install keyboard listener
            pipeline.install([&perf_monitor](Pipeline &pipeline, int key) {
                switch (key) {
                case 'p':
                    pipeline.pause();
                    perf_monitor.pause();
                    break;
                case 'r':
                    pipeline.resume();
                    perf_monitor.resume();
                    break;
                case 'q':
                    pipeline.stop();
                    break;
                default:
                    break;
                }
            });

            // start pipeline with message monitor
            pipeline.start([&otf_trigger, &tiler_event_handler, &sensor_map, &perf_monitor](
                               Pipeline &p, const Pipeline::Message &msg) {
                const Pipeline::StateTransitionMessage *state_change_msg =
                    dynamic_cast<const Pipeline::StateTransitionMessage *>(&msg);
                const Pipeline::DynamicSourceMessage *source_change_msg =
                    dynamic_cast<const Pipeline::DynamicSourceMessage *>(&msg);
                // handle state change message
                if (state_change_msg) {
                    static bool tiler_event_handler_started = false;
                    if (!tiler_event_handler_started) {
                        Pipeline::State old_state = Pipeline::State::EMPTY;
                        Pipeline::State new_state = Pipeline::State::EMPTY;
                        state_change_msg->getState(old_state, new_state);
                        if (new_state == Pipeline::State::PLAYING &&
                            state_change_msg->getName() == "sink") {
                            tiler_event_handler.start();
                            tiler_event_handler_started = true;
                        }
                    }
                    static bool otf_trigger_started = false;
                    if (!otf_trigger_started) {
                        otf_trigger.start();
                        otf_trigger_started = true;
                    }
                }

                // handle source add/remove message
                if (source_change_msg) {
                    auto source_id = source_change_msg->getSourceId();
                    if (source_change_msg->isSourceAdded()) {
                        auto uri = source_change_msg->getUri();
                        auto sensor_id = source_change_msg->getSensorId();
                        auto sensor_name = source_change_msg->getSensorName();
                        if (sensor_map.find(source_id) == sensor_map.end()) {
                            sensor_map[source_id] = SensorInfo{uri, sensor_id, sensor_name};
                        } else {
                            std::cout << "Warning, sensor with source id " << source_id
                                      << " already added" << std::endl;
                        }
                        perf_monitor.addStream(source_id, uri.c_str(), sensor_id.c_str(),
                                               sensor_name.c_str());
                    } else {
                        sensor_map.erase(source_id);
                        perf_monitor.removeStream(source_id);
                    }
                }
            });
        }
        // wait for all the pipelines to end
        for (auto &pipeline : pipelines) {
            pipeline->wait();
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
