/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */
/**
 * @file
 * <b>Defines Tracker Metadata</b>
 */
/**
 * @defgroup  ee_tracker_group Tracker Metadata
 *
 * Specifies metadata concerning tracking.
 *
 * @ingroup NvDsMetaApi
 * @{
 */
#ifndef _NVDS_TRACKER_META_H_
#define _NVDS_TRACKER_META_H_

#include <stdint.h>

#include "nvdsmeta.h"
#include "nvll_osd_struct.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * One target in a single past frame
 */
typedef struct _NvDsPastFrameObj {
    /** Frame number. */
    uint32_t frameNum;
    /** Bounding box. */
    NvOSD_RectParams tBbox;
    /** Tracking confidence. */
    float confidence;
    /** Tracking age. */
    uint32_t age;
} NvDsPastFrameObj;

/**
 * One target in several past frames
 */
typedef struct _NvDsPastFrameObjList {
    /** Pointer to past frame info of this target. */
    NvDsPastFrameObj *list;
    /** Number of frames this target appreared in the past. */
    uint32_t numObj;
    /** Target tracking id. */
    uint64_t uniqueId;
    /** Target class id. */
    uint16_t classId;
    /** An array of the string describing the target class. */
    gchar objLabel[MAX_LABEL_SIZE];
} NvDsPastFrameObjList;

/**
 * List of targets in each stream
 */
typedef struct _NvDsPastFrameObjStream {
    /** Pointer to targets inside this stream. */
    NvDsPastFrameObjList *list;
    /** Stream id the same as frame_meta->pad_index. */
    uint32_t streamID;
    /** Stream id used inside tracker plugin. */
    uint64_t surfaceStreamID;
    /** Maximum number of objects allocated. */
    uint32_t numAllocated;
    /** Number of objects in this frame. */
    uint32_t numFilled;
} NvDsPastFrameObjStream;

/**
 * Batch of past frame targets in all streams
 */
typedef struct _NvDsPastFrameObjBatch {
    /** Pointer to array of stream lists. */
    NvDsPastFrameObjStream *list;
    /** Number of blocks allocated for the list. */
    uint32_t numAllocated;
    /** Number of filled blocks in the list. */
    uint32_t numFilled;
    /** Pointer to internal buffer pool needed by gst pipelines to return buffers. */
    void *priv_data;
} NvDsPastFrameObjBatch;

/**
 * ReID tensor of the batch.
 */
typedef struct _NvDsReidTensorBatch {
    /** Each target's ReID vector length. */
    uint32_t featureSize;
    /** Number of reid vectors in the batch. */
    uint32_t numFilled;
    /** ReID vector on CPU. */
    float *ptr_host;
    /** ReID vector on GPU. */
    float *ptr_dev;
    /** Pointer to internal buffer pool needed by gst pipelines to return buffers.*/
    void *priv_data;
} NvDsReidTensorBatch;

/**
 * Batch of trajectory data in all streams.
 */
typedef NvDsPastFrameObjBatch NvDsTrajectoryBatch;

#ifdef __cplusplus
}
#endif

#endif

/** @} */
