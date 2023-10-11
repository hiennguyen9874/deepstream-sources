################################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

# This script performs int8 calibration for the re-identification model. It loads
# network information from tracker config. The  generated int8 calibration
# file and engine can be used for tracker ReID model to maximize performance.
# The script can be modified to perform calibration for other networks.

import os
import sys
import numpy as np
import tensorrt as trt
from PIL import Image
import yaml

import pycuda.driver as cuda
import pycuda.autoinit

def load_reid_input(img_dir):
    """Load reid image patches for calibration"""
    file_list = os.listdir(img_dir)
    img_list = []
    for file in file_list:
        if file.endswith(".png") or file.endswith(".jpg"):
            img = Image.open(os.path.join(img_dir, file))
            img_list.append(np.array(img, dtype=np.int32))
    if len(img_list) == 0:
        raise Exception("No calibration images provided. Place them under {}".format(img_dir))
    return np.stack(img_list).astype(np.float32)

class ReIDEntropyCalibrator(trt.IInt8EntropyCalibrator):
    """ReID calibrator class"""
    def __init__(self, infer_dims, cache_file, data_dir):
        # Call the constructor of the parent explicitly
        trt.IInt8EntropyCalibrator.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = load_reid_input(data_dir)
        self.batch_size = 4
        self.infer_dims = infer_dims
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.infer_dims[0] * self.infer_dims[1] * self.infer_dims[2] * self.batch_size * 4)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images...".format(current_batch, self.batch_size))

        batch = np.ascontiguousarray(self.data[self.current_index: self.current_index + self.batch_size])
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_engine(reid_config, data_dir):
    """Perform calibration and build engine"""
    trt_logger = trt.Logger()
    with trt.Builder(
        trt_logger
    ) as builder, builder.create_network() as network, builder.create_builder_config() as builder_config, trt.UffParser() as parser, trt.Runtime(
        trt_logger
    ) as runtime:

        # Load network info from tracjer config
        cache_file = "../../samples/models/Tracker/calibration.cache"
        if "calibrationTableFile" in reid_config:
            cache_file = reid_config["calibrationTableFile"]
        else:
            print("calibrationTableFile not found in tracker config, using {} as default output".format(cache_file))

        if reid_config["inputOrder"] == "0":
            input_order = trt.UffInputOrder.NCHW
        elif reid_config["inputOrder"] == "1":
            input_order = trt.UffInputOrder.NHWC
        else:
            raise Exception("Invalid input order")

        infer_dims = [int(d) for d in reid_config["inferDims"]]

        # Create engine builder config and uff parser
        builder_config.max_workspace_size = int(reid_config["workspaceSize"]) << 20
        builder_config.set_flag(trt.BuilderFlag.INT8)
        builder_config.int8_calibrator = ReIDEntropyCalibrator(infer_dims, cache_file, data_dir)
        builder.max_batch_size = int(reid_config["batchSize"])

        parser.register_input(reid_config["inputBlobName"], infer_dims, input_order)
        parser.register_output(reid_config["outputBlobName"])
        parser.parse(reid_config["uffFile"], network)

        # Perform calibration and create engine
        print("Generating calibration file and engine (It's OK to miss scale and zero-point for some tensors)...")
        engine = builder.build_serialized_network(network, builder_config)
        print("Saved calibration table file to {}".format(cache_file))
        engine_file = "{}_b{}_gpu0_int8.engine".format(reid_config["uffFile"], builder.max_batch_size)
        with open(engine_file, "wb") as f:
            f.write(engine)
            print("Saved engine file to {}".format(engine_file))


if __name__ == "__main__":
    config_file = "../../samples/configs/deepstream-app/config_tracker_NvDeepSORT.yml" # or config_tracker_NvDCF_accuracy.yml
    data_dir = "./data/"

    if len(sys.argv) < 3:
        print('Usage: python3 ' + sys.argv[0] + ' /path/to/data_dir/ /path/to/config_tracker.yml')
        print("Using config_file=\"{}\", data_dir=\"{}\" by default".format(config_file, data_dir))

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        config_file = sys.argv[2]

    with open(config_file, "r") as f:
        lines = "".join(f.readlines()[1:])
        reid_config = yaml.load(lines, yaml.BaseLoader)["ReID"]

    build_engine(reid_config, data_dir)