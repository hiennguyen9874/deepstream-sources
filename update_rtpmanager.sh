#!/bin/bash
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during installation
export DEBIAN_FRONTEND=noninteractive

#Install prerequisites
apt-get -y install python3-pip
pip3 install meson --break-system-packages
pip3 install ninja --break-system-packages
pip3 install setuptools==73.0.0 --break-system-packages
apt-get -y install libmount-dev
apt-get -y install flex
apt-get -y install flex bison
apt-get -y install libharfbuzz-dev
apt-get -y install libpango1.0-dev

if [ -d "/tmp/gst-1.24.2/" ]; then
    rm -rf /tmp/gst-1.24.2/
fi

#Build gstreamer and place the rtpmanager library at the required path.
mkdir -p /tmp/gst-1.24.2
pushd /tmp/gst-1.24.2
git clone https://gitlab.freedesktop.org/gstreamer/gstreamer.git -b 1.24.2
pushd gstreamer
#Place rtpjitterbuffer_eos_handling.patch at the appropriate location
git apply /opt/nvidia/deepstream/deepstream/rtpjitterbuffer_eos_handling.patch
#Build gstreamer 1,20.3 with the applied fix patch
meson build --buildtype=release -Dbad=disabled -Dugly=disabled -Dexamples=disabled -Dlibav=disabled -Ddevtools=disabled
ninja -C build/
#Place the rtpmanager library at the gstreamer installation path
if [[ $(uname -m) == "aarch64" ]]; then
    cp build/subprojects/gst-plugins-good/gst/rtpmanager/libgstrtpmanager.so /usr/lib/aarch64-linux-gnu/gstreamer-1.0
else
    cp build/subprojects/gst-plugins-good/gst/rtpmanager/libgstrtpmanager.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0
fi

popd
popd
rm -rf /tmp/gst-1.24.2
