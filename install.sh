################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

#!/bin/bash

TARGET_DEVICE=$(uname -m)
OS=$(cat /etc/os-release | awk -F= '$1=="ID"{print $2}' | sed 's/"//g')

if [ "${TARGET_DEVICE}" = "x86_64" ]; then
    if [ "${OS}" = "rhel" ]; then
        mkdir -p /usr/lib/x86_64-linux-gnu/libv4l/plugins/
        ln -sf /opt/nvidia/deepstream/deepstream-8.0/lib/libv4l/plugins/libcuvidv4l2_plugin.so /usr/lib/x86_64-linux-gnu/libv4l/plugins/libcuvidv4l2_plugin.so
        BASE_LIB_DIR="/usr/lib64/"
    elif [ "${OS}" = "ubuntu" ]; then
        BASE_LIB_DIR="/usr/lib/x86_64-linux-gnu/"
    else
        echo "Unsupported OS" 2>&1
        exit 1
    fi
elif [ "${TARGET_DEVICE}" = "aarch64" ]; then
    BASE_LIB_DIR="/usr/lib/aarch64-linux-gnu"
    if [ -d "$BASE_LIB_DIR/nvidia" ]; then
        NVIDIA_LIB_DIR="$BASE_LIB_DIR/nvidia"
    else
        NVIDIA_LIB_DIR="$BASE_LIB_DIR/tegra"
    fi
fi

while getopts "d:" option; do
    case "${option}" in
    d)
        BASE_LIB_DIR="${OPTARG}"
        ;;
    *)
        echo "ERROR! Unsupported option!"
        exit 1
        ;;
    esac
done

if [ -n $TARGET_DEVICE ]; then
    if [ "${TARGET_DEVICE}" = "x86_64" ]; then
        update-alternatives --install $BASE_LIB_DIR/gstreamer-1.0/deepstream deepstream-plugins /opt/nvidia/deepstream/deepstream-8.0/lib/gst-plugins 80
        update-alternatives --install $BASE_LIB_DIR/libv4l/plugins/libcuvidv4l2_plugin.so deepstream-v4l2plugin /opt/nvidia/deepstream/deepstream-8.0/lib/libv4l/plugins/libcuvidv4l2_plugin.so 80
        update-alternatives --install /usr/bin/deepstream-appsrc-cuda-test deepstream-appsrc-cuda-test /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-appsrc-cuda-test 80
        update-alternatives --install /usr/bin/deepstream-avsync-app deepstream-avsync-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-avsync-app 80
        update-alternatives --install /usr/bin/deepstream-multigpu-nvlink-test deepstream-multigpu-nvlink-test /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-multigpu-nvlink-test 80
        update-alternatives --install $BASE_LIB_DIR/libv4l2.so.0.0.99999 deepstream-v4l2library /opt/nvidia/deepstream/deepstream-8.0/lib/libnvv4l2.so 80
        update-alternatives --install $BASE_LIB_DIR/libv4lconvert.so.0.0.99999 deepstream-v4lconvert /opt/nvidia/deepstream/deepstream-8.0/lib/libnvv4lconvert.so 80
	    update-alternatives --install /usr/bin/deepstream-ucx-test-app deepstream-ucx-test-app-client /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-ucx-test-app 80
    elif [ "${TARGET_DEVICE}" = "aarch64" ]; then
        update-alternatives --install $BASE_LIB_DIR/gstreamer-1.0/deepstream deepstream-plugins /opt/nvidia/deepstream/deepstream-8.0/lib/gst-plugins 80
        ln -sf $NVIDIA_LIB_DIR/libnvbufsurface.so /opt/nvidia/deepstream/deepstream-8.0/lib/libnvbufsurface.so
        ln -sf $NVIDIA_LIB_DIR/libnvbufsurftransform.so /opt/nvidia/deepstream/deepstream-8.0/lib/libnvbufsurftransform.so
        ln -sf $NVIDIA_LIB_DIR/libnvdsbufferpool.so /opt/nvidia/deepstream/deepstream-8.0/lib/libnvdsbufferpool.so
        ln -sf $NVIDIA_LIB_DIR/libgstnvcustomhelper.so /opt/nvidia/deepstream/deepstream-8.0/lib/libgstnvcustomhelper.so
        update-alternatives --install /usr/bin/deepstream-ipc-test-app deepstream-ipc-test-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-ipc-test-app 80
        echo "/opt/nvidia/deepstream/deepstream-8.0/lib" > /etc/ld.so.conf.d/deepstream.conf
        echo "/opt/nvidia/deepstream/deepstream-8.0/lib/gst-plugins" >> /etc/ld.so.conf.d/deepstream.conf
    fi
fi
update-alternatives --install /usr/bin/deepstream-app deepstream-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-app 80
update-alternatives --install /usr/bin/deepstream-audio deepstream-audio /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-audio 80
update-alternatives --install /usr/bin/deepstream-asr-app deepstream-asr-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-asr-app 80
update-alternatives --install /usr/bin/deepstream-asr-tts-app deepstream-asr-tts-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-asr-tts-app 80
update-alternatives --install /usr/bin/deepstream-test1-app deepstream-test1-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-test1-app 80
update-alternatives --install /usr/bin/deepstream-test2-app deepstream-test2-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-test2-app 80
update-alternatives --install /usr/bin/deepstream-test3-app deepstream-test3-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-test3-app 80
update-alternatives --install /usr/bin/deepstream-test4-app deepstream-test4-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-test4-app 80
update-alternatives --install /usr/bin/deepstream-test5-app deepstream-test5-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-test5-app 80
update-alternatives --install /usr/bin/deepstream-testsr-app deepstream-testsr-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-testsr-app 80
update-alternatives --install /usr/bin/deepstream-transfer-learning-app deepstream-transfer-learning-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-transfer-learning-app 80
update-alternatives --install /usr/bin/deepstream-user-metadata-app deepstream-user-metadata-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-user-metadata-app 80
update-alternatives --install /usr/bin/deepstream-dewarper-app deepstream-dewarper-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-dewarper-app 80
update-alternatives --install /usr/bin/deepstream-nvof-app deepstream-nvof-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-nvof-app 80
update-alternatives --install /usr/bin/deepstream-image-decode-app deepstream-image-decode-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-image-decode-app 80
update-alternatives --install /usr/bin/deepstream-gst-metadata-app deepstream-gst-metadata-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-gst-metadata-app 80
update-alternatives --install /usr/bin/deepstream-opencv-test deepstream-opencv-test /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-opencv-test 80
update-alternatives --install /usr/bin/deepstream-preprocess-test deepstream-preprocess-test /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-preprocess-test 80
update-alternatives --install /usr/bin/deepstream-image-meta-test deepstream-image-meta-test /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-image-meta-test 80
update-alternatives --install /usr/bin/deepstream-appsrc-test deepstream-appsrc-test /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-appsrc-test 80
update-alternatives --install /usr/bin/deepstream-can-orientation-app deepstream-can-orientation-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-can-orientation-app 80
update-alternatives --install /usr/bin/deepstream-nvdsanalytics-test deepstream-nvdsanalytics-test /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-nvdsanalytics-test 80
update-alternatives --install /usr/bin/deepstream-3d-action-recognition deepstream-3d-action-recognition /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-3d-action-recognition 80
update-alternatives --install /usr/bin/deepstream-3d-depth-camera deepstream-3d-depth-camera /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-3d-depth-camera 80
update-alternatives --install /usr/bin/deepstream-lidar-inference-app deepstream-lidar-inference-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-lidar-inference-app 80
update-alternatives --install /usr/bin/deepstream-3d-lidar-sensor-fusion deepstream-3d-lidar-sensor-fusion /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-3d-lidar-sensor-fusion 80
update-alternatives --install /usr/bin/deepstream-nmos-app deepstream-nmos-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-nmos-app 80
update-alternatives --install /usr/bin/deepstream-server-app deepstream-server-app /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-server-app 80
update-alternatives --install /usr/bin/deepstream-demuxer-static deepstream-demuxer-static /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-demuxer-static 80
update-alternatives --install /usr/bin/deepstream-demuxer-dynamic deepstream-demuxer-dynamic /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-demuxer-dynamic 80
update-alternatives --install /usr/bin/service-maker-3d-action-recognition-app service-maker-3d-action-recognition-app /opt/nvidia/deepstream/deepstream-8.0/bin/service-maker-3d-action-recognition-app 80
update-alternatives --install /usr/bin/service-maker-appsrc-test-app service-maker-appsrc-test-app /opt/nvidia/deepstream/deepstream-8.0/bin/service-maker-appsrc-test-app 80
update-alternatives --install /usr/bin/service-maker-sr-test-app service-maker-sr-test-app /opt/nvidia/deepstream/deepstream-8.0/bin/service-maker-sr-test-app 80
update-alternatives --install /usr/bin/service-maker-test1-app service-maker-test1-app /opt/nvidia/deepstream/deepstream-8.0/bin/service-maker-test1-app 80
update-alternatives --install /usr/bin/service-maker-test2-app service-maker-test2-app /opt/nvidia/deepstream/deepstream-8.0/bin/service-maker-test2-app 80
update-alternatives --install /usr/bin/service-maker-test3-app service-maker-test3-app /opt/nvidia/deepstream/deepstream-8.0/bin/service-maker-test3-app 80
update-alternatives --install /usr/bin/service-maker-test4-app service-maker-test4-app /opt/nvidia/deepstream/deepstream-8.0/bin/service-maker-test4-app 80
update-alternatives --install /usr/bin/service-maker-test5-app service-maker-test5-app /opt/nvidia/deepstream/deepstream-8.0/bin/service-maker-test5-app 80
pip install /opt/nvidia/deepstream/deepstream-8.0/service-maker/python/pyservicemaker*.whl --force-reinstall --break-system-packages
ldconfig
rm -rf /home/*/.cache/gstreamer-1.0/
rm -rf /root/.cache/gstreamer-1.0/
