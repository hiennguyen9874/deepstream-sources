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

from pyservicemaker import Pipeline, Flow, BatchMetadataOperator, Probe, osd, SmartRecordConfig
from multiprocessing import Process
import sys

CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml"

KAFKA_PROTO_LIB_PATH = "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
KAFKA_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/sources/libs/kafka_protocol_adaptor/cfg_kafka.txt"
MSGCONV_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test5/configs/dstest5_msgconv_sample_config.txt"
KAFKA_CONN_STR = "localhost;9092"
KAFKA_TOPIC_LIST = "sr-test"

MUXER_WIDTH = 1920
MUXER_HEIGHT = 1080

def deepstream_sr_test_app(stream_file_path_list):
    pipeline = Pipeline("deepstream-sr-test")
    smart_record_config = SmartRecordConfig(
        proto_lib=KAFKA_PROTO_LIB_PATH,
        conn_str=KAFKA_CONN_STR,
        msgconv_config_file=MSGCONV_CONFIG_FILE,
        proto_config_file=KAFKA_CONFIG_FILE,
        topic_list=KAFKA_TOPIC_LIST,
        smart_rec_cache=20,
        smart_rec_container=0,
        smart_rec_dir_path=".",
        smart_rec_mode=0
    )
    flow = Flow(pipeline).batch_capture(
        stream_file_path_list,
        smart_record_config=smart_record_config,
        width=MUXER_WIDTH,
        height=MUXER_HEIGHT,
    ).infer(CONFIG_FILE_PATH, batch_size=len(stream_file_path_list)).render(sync=False)()

if __name__ == '__main__':
    # Check input arguments
    if len(sys.argv) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN]\n" % sys.argv[0])
        sys.exit(1)

    # Flow()() is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=deepstream_sr_test_app, args=(sys.argv[1:],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()