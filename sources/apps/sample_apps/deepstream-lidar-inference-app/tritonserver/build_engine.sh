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
#!/bin/bash

IS_JETSON_PLATFORM=`uname -i | grep aarch64`

export PATH=$PATH:/usr/src/tensorrt/bin


if [ ! ${IS_JETSON_PLATFORM} ]; then
    wget --no-check-certificate --content-disposition 'https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v3.22.05_trt8.4_x86/files/tao-converter' -O tao-converter
else
    wget --no-check-certificate --content-disposition 'https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v3.22.05_trt8.4_aarch64/files/tao-converter' -O tao-converter
fi
chmod 755 tao-converter

mkdir -p models/pointpillars/1/
wget --no-check-certificate 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/pointpillarnet/versions/deployable_v1.0/files/pointpillars_deployable.etlt' -O ./models/pointpillars/1/pointpillars_deployable.etlt
wget --no-check-certificate 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/pointpillarnet/versions/deployable_v1.0/files/label.txt' -O ./models/pointpillars/labels.txt

./tao-converter  -k tlt_encode -e ./models/pointpillars/1/trt.fp32.engine -p points,1x204800x4,1x204800x4,1x204800x4  -p num_points,1,1,1  -t fp32 ./models/pointpillars/1/pointpillars_deployable.etlt
