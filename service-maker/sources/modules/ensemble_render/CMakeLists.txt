####################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
####################################################################################################

cmake_minimum_required(VERSION 3.16)

project(Sample)
find_package(nvds_service_maker REQUIRED PATHS /opt/nvidia/deepstream/deepstream-8.0/service-maker/cmake)
find_package(CUDA)

set(DS_PATH "/opt/nvidia/deepstream/deepstream-8.0")
add_library(ensemble_render SHARED ensemble_render.cpp)
# Add include directories for cuda
target_include_directories(ensemble_render PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(ensemble_render PRIVATE ${DS_PATH}/sources/includes)
target_include_directories(ensemble_render PRIVATE ${DS_PATH}/sources/includes/ds3d/commmon)
target_include_directories(ensemble_render PRIVATE ${DS_PATH}/sources/libs)
include_directories(
    /usr/include/gstreamer-1.0
    /usr/include/glib-2.0
    /usr/lib/aarch64-linux-gnu/glib-2.0/include/
    /usr/lib/x86_64-linux-gnu/glib-2.0/include
    /usr/local/cuda/include
)
target_link_libraries(ensemble_render nvds_service_maker nvds_service_maker_utils)
target_link_libraries(ensemble_render ${DS_PATH}/lib/libnvds_3d_common.so)
target_link_libraries(ensemble_render ${DS_PATH}/lib/libnvdsgst_3d_gst.so)
