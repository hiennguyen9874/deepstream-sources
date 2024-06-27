/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <sys/stat.h>

#include <fstream>
#include <string>

#include "buffer_probe.hpp"

#define UNTRACKED_OBJECT_ID 0xFFFFFFFFFFFFFFFF

namespace deepstream {

class NvDsKittiDump : public BufferProbe::IBatchMetadataOperator {
public:
    NvDsKittiDump() {}
    virtual probeReturn handleData(BufferProbe &probe, BatchMetadata &data)
    {
        char bbox_file[1024] = {0};
        std::string output_dir = "/tmp/kitti/";
        probe.getProperty("kitti-dir", output_dir);
        FILE *bbox_params_dump_file = NULL;
        // if (!buffer_data.get<NvDsBatchMetaHandle>(BUF_DATA_KEY_VIDEO_BATCH_META)) {
        //     return true;
        // }

        FrameMetadata::Iterator frame_itr;

        for (data.initiateIterator(frame_itr); !frame_itr->done(); frame_itr->next()) {
            // guint stream_id = frame_meta->pad_index;
            uint stream_id = (*frame_itr)->padIndex();
            snprintf(bbox_file, sizeof(bbox_file) - 1, "%s/%03u_%06lu.txt", output_dir.c_str(),
                     stream_id, (ulong)(*frame_itr)->frameNum());
            bbox_params_dump_file = fopen(bbox_file, "w");
            if (!bbox_params_dump_file)
                continue;

            FrameMetadata &frame_meta = frame_itr->get();
            ObjectMetadata::Iterator obj_itr;

            for (frame_meta.initiateIterator(obj_itr); !obj_itr->done(); obj_itr->next()) {
                ObjectMetadata &object_meta = obj_itr->get();
                float confidence = object_meta.confidence();
                float left = object_meta.rectParams().left;
                float top = object_meta.rectParams().top;
                float right = left + object_meta.rectParams().width;
                float bottom = top + object_meta.rectParams().height;

                if (object_meta.objectId() == UNTRACKED_OBJECT_ID) {
                    fprintf(bbox_params_dump_file,
                            "%s 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
                            object_meta.label().c_str(), left, top, right, bottom, confidence);
                } else {
                    fprintf(bbox_params_dump_file,
                            "%s %lu 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
                            object_meta.label().c_str(), object_meta.objectId(), left, top, right,
                            bottom, confidence);
                }
            }
            fclose(bbox_params_dump_file);
        }
        return probeReturn::Probe_Ok;
    }
};

} // namespace deepstream
