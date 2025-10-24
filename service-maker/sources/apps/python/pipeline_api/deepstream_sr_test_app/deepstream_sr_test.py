#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator, osd, CommonFactory, signal
from multiprocessing import Process
import os
import sys
import platform

PIPELINE_NAME = "deepstream-sr-test"

KAFKA_PROTO_LIB_PATH = "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
KAFKA_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/sources/libs/kafka_protocol_adaptor/cfg_kafka.txt"
MSGCONV_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test5/configs/dstest5_msgconv_sample_config.txt"
KAFKA_CONN_STR = "localhost;9092"
KAFKA_TOPIC_LIST = "sr-test"

CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml"

BATCHED_PUSH_TIMEOUT = 33000
MUXER_WIDTH = 1920
MUXER_HEIGHT = 1080
TILER_WIDTH = 1280
TILER_HEIGHT = 720


def main(file_path):
  file_list = file_path if isinstance(file_path, list) else [file_path]
  num_sources = len(file_list)

  pipeline = Pipeline(PIPELINE_NAME)
  pipeline.add("nvstreammux", "mux", {"batch-size": num_sources, "batched-push-timeout": BATCHED_PUSH_TIMEOUT, "width": MUXER_WIDTH, "height": MUXER_HEIGHT})

  sr_controller = CommonFactory.create("smart_recording_action", "sr_controller")
  if isinstance(sr_controller, signal.Emitter):
    sr_controller_properties = {
            "proto-lib": KAFKA_PROTO_LIB_PATH,
            "conn-str": KAFKA_CONN_STR,
            "msgconv-config-file": MSGCONV_CONFIG_FILE,
            "proto-config-file": KAFKA_CONFIG_FILE,
            "topic-list": KAFKA_TOPIC_LIST
    }
    sr_controller.set(sr_controller_properties)

    for i, file in enumerate(file_list):
      source_name = f"src_{i}"
      pipeline.add("nvurisrcbin", source_name, {"uri": file, "smart-record": 1, "smart-rec-cache": 20, "smart-rec-container": 0, "smart-rec-dir-path": ".", "smart-rec-mode": 0})
      pipeline.link((source_name, "mux"), ("", "sink_%u"))
      sr_controller.attach("start-sr", pipeline[source_name])
      sr_controller.attach("stop-sr", pipeline[source_name])
      pipeline.attach(source_name, "smart_recording_signal", "sr", "sr-done")

    pipeline.add("nvinfer", "infer", {"config-file-path": CONFIG_FILE_PATH, "batch-size": len(file_list)})
    pipeline.add("nvmultistreamtiler", "tiler", {"width": TILER_WIDTH, "height": TILER_HEIGHT})
    pipeline.add("nvosdbin", "osd").add("nv3dsink" if platform.processor() == "aarch64" else "nveglglessink", "sink", {"sync": False})
    pipeline.link("mux", "infer", "tiler", "osd", "sink")

    pipeline.start().wait()

if __name__ == '__main__':
    # Check input arguments
    if len(sys.argv) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN]\n" % sys.argv[0])
        sys.exit(1)
    # pipeline.wait() in the main function is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=main, args=(sys.argv[1:],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()