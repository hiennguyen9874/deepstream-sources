################################################################################
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

# This script convert DeepSORT's official re-identification model from TensorFlow
# frozen graph into UFF. The node edition is specific to the official model, since
# it deletes nodes not suppported by TensorRT.

import os
import graphsurgeon as gs
import tensorflow as tf
import uff
import sys
import yaml

def convert_to_uff(filename_pb, filename_uff):
    dynamic_graph = gs.DynamicGraph(filename_pb)
    nodes = list(dynamic_graph.as_graph_def().node)

    print('Converting...')
    # create input node
    input_node = gs.create_node("images",
            op="Placeholder",
            dtype=tf.float32,
            shape=[None, 128, 64, 3]
            )

    # remove nodes in DeepSORT's re-identification model not supported by TensorRT,
    # and connect with input node
    for node in nodes:
        if "map" in node.name or "images" == node.name or "Cast" == node.name:
            dynamic_graph.remove(node)
        elif "conv1_1/Conv2D" == node.name:
            node.input.insert(0, "images")

    # add input node to graph
    dynamic_graph.append(input_node)

    # create uff file
    trt_graph = uff.from_tensorflow(dynamic_graph.as_graph_def())
    #filename_uff = filename_pb[:filename_pb.rfind('.')]  + '.uff'
    print('Writing to disk...')
    with open(filename_uff, 'wb') as f:
        f.write(trt_graph)
    print('Saved as ' + filename_uff)

if __name__ == "__main__":
    config_file = "../../samples/configs/deepstream-app/config_tracker_NvDeepSORT.yml" # or config_tracker_NvDCF_accuracy.yml
    filename_pb = "mars-small128.pb"

    if len(sys.argv) < 3:
        print('Usage: python3 ' + sys.argv[0] + ' /path/to/model.pb')
        print("Using filename_pb=\"{}\", config_file=\"{}\" by default".format(filename_pb, config_file))

    if len(sys.argv) > 1:
        filename_pb = sys.argv[1]

    if len(sys.argv) > 2:
        config_file = sys.argv[2]

    with open(config_file, "r") as f:
        lines = "".join(f.readlines()[1:])
        reid_config = yaml.load(lines, yaml.BaseLoader)["ReID"]

    filename_uff = filename_pb[:filename_pb.rfind('.')]  + '.uff'
    if "uffFile" in reid_config:
        filename_uff = reid_config["uffFile"]
        dir_uff = os.path.dirname(filename_uff)
        if not os.path.isdir(dir_uff):
            print("Making directory {}".format(dir_uff))
            os.makedirs(dir_uff)

    convert_to_uff(filename_pb, filename_uff)
