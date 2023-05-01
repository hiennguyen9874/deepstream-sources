/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _DS3D_COMMON_LIDAR_DATATYPE__H
#define _DS3D_COMMON_LIDAR_DATATYPE__H

#include <ds3d/common/common.h>
#include <ds3d/common/idatatype.h>

#undef DS3D_MAX_LABEL_SIZE
#define DS3D_MAX_LABEL_SIZE 128

/*
 * Pointcloud Coordinate System
 * pointcloud & box coordinate system, x -> front, y -> left, z -> up
 *
 *
 *                                up z    x front (yaw=0)
 *                                   ^   ^
 *                                   |  /
 *                                   | /
 *       (yaw=0.5*pi) left y <------ 0
 */
namespace ds3d {

struct Lidar3DBbox {
    float centerX = 0.0f;
    float centerY = 0.0f;
    float centerZ = 0.0f;
    float width = 0.0f;
    float length = 0.0f;
    float height = 0.0f;
    float yaw = 0.0f;
    int cid = 0;
    float score = 0.0f;
    vec4b bboxColor = {{255, 255, 0, 255}}; // RGBA
    char labels[DS3D_MAX_LABEL_SIZE] = {0};

    Lidar3DBbox() = default;
    Lidar3DBbox(float centerX_,
                float centerY_,
                float centerZ_,
                float width_,
                float length_,
                float height_,
                float yaw_,
                int cid_,
                float score_)
        : centerX(centerX_), centerY(centerY_), centerZ(centerZ_), width(width_), length(length_),
          height(height_), yaw(yaw_), cid(cid_), score(score_)
    {
    }
};

} // namespace ds3d

#endif