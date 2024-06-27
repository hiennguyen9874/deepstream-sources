#!/usr/bin/env bash

################################################################################
# Copyright (c) 2019-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

# Download yolo weights
# For yolo v2,
echo "Downloading yolov2 config and weights files ... "
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg -q --show-progress
wget https://pjreddie.com/media/files/yolov2.weights -q --show-progress

# For yolo v2 tiny,
echo "Downloading yolov2-tiny config and weights files ... "
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny.cfg -q --show-progress
wget https://pjreddie.com/media/files/yolov2-tiny.weights -q --show-progress

# For yolo v3,
echo "Downloading yolov3 config and weights files ... "
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -q --show-progress
wget https://pjreddie.com/media/files/yolov3.weights -q --show-progress

# For yolo v3 tiny,
echo "Downloading yolov3-tiny config and weights files ... "
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg -q --show-progress
wget https://pjreddie.com/media/files/yolov3-tiny.weights -q --show-progress
