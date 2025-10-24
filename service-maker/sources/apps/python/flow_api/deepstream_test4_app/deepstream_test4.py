#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from pyservicemaker import Pipeline, Flow, BatchMetadataOperator, Probe, osd
from multiprocessing import Process
import sys

CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test4/dstest4_pgie_config.yml"
CONFIG_MSGCONV_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test4/dstest4_msgconv_config.yml"
KAFKA_PROTO_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
MSGBROKER_CONN_STR = "localhost;9092"
MGSBROKER_TOPIC = "test4app"


def deepstream_test4_app(stream_file_path):
    pipeline = Pipeline("deepstream-test4")
    flow = Flow(pipeline).batch_capture([stream_file_path]).infer(CONFIG_FILE_PATH)
    flow = flow.attach(
            what="add_message_meta_probe",
            name="message_generator"
        ).fork()
    flow.publish(
            msg_broker_proto_lib=KAFKA_PROTO_LIB,
            msg_broker_conn_str=MSGBROKER_CONN_STR,
            topic=MGSBROKER_TOPIC,
            msg_conv_config=CONFIG_MSGCONV_PATH
        )
    flow.render()
    flow()
    

if __name__ == '__main__':
    # Check input arguments
    if len(sys.argv) != 2:
        sys.stderr.write("usage: %s <H264 filename> \n" % sys.argv[0])
        sys.exit(1)

    # Flow()() is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=deepstream_test4_app, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()