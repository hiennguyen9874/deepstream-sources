/**
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _INVTRACKERPROC_H
#define _INVTRACKERPROC_H

#include <vector>

#include "nvbufsurface.h"
#include "nvdstracker.h"

/** Input data for tracker plugin. */
struct InputParams {
    NvBufSurface *pSurfaceBatch;
    NvDsBatchMeta *pBatchMeta;
    void *pPreservedData;
    bool eventMarker;
};

/** Tracker process completion status. */
enum CompletionStatus { CompletionStatus_OK, CompletionStatus_Error, CompletionStatus_Exit };

/** Bitwise flags for tracker id reset. */
enum TrackingIdResetMode {
    /** No id reset. */
    TrackingIdResetMode_Default = 0,
    /** Terminate existing objects and assign new ids after stream reset. */
    TrackingIdResetMode_NewIdAfterStreamReset = 1 << 0,
    /** Id starts from 0 after stream reaching EOS. */
    TrackingIdResetMode_FromZeroAfterEOS = 1 << 1,
    /** Max value of the enum. Supporting both NewIdAfterStreamReset and FromZeroAfterEOS. */
    TrackingIdResetMode_MaxValue = 3
};

/** Tracker plugin config params. */
struct TrackerConfig {
    /** From DeepStream app config file. */
    uint32_t batchSize;
    uint32_t trackerWidth;
    uint32_t trackerHeight;
    char *trackerLibFile;
    char *trackerConfigFile;

    bool displayTrackingId;
    TrackingIdResetMode trackingIdResetMode;

    NvMOTCompute computeTarget;
    int32_t gpuId;
    int32_t compute_hw;

    uint32_t trackingSurfType;
    bool trackingSurfTypeFromConfig = false;

    bool inputTensorMeta = false;
    uint32_t tensorMetaGieId = 0;

    /** From low level tracker library query. */
    NvBufSurfaceColorFormat colorFormat;
    NvBufSurfaceMemType memType;
    uint32_t numTransforms;
    uint32_t maxTargetsPerStream;
    uint32_t maxShadowTrackingAge;
    bool pastFrame;

    /** Store buffer pool size since low level tracker needs this info. */
    uint32_t maxConvBufPoolSize;
    uint32_t maxMiscDataPoolSize;
    uint32_t reidFeatureSize;
    bool outputReidTensor;

    char *gstName;
};

/** Virtual base class for tracker plugin processing. */
class INvTrackerProc {
public:
    virtual ~INvTrackerProc(){};

    virtual bool init(const TrackerConfig &config) = 0;
    virtual void deInit() = 0;

    /** Tracker actions when a source is added to the pipeline. */
    virtual bool addSource(uint32_t sourceId) = 0;
    /** Tracker actions when a source is removed to the pipeline. */
    virtual bool removeSource(uint32_t sourceId, bool removeObjectIdMapping = true) = 0;
    /** Tracker actions when a source is reset. */
    virtual bool resetSource(uint32_t sourceId) = 0;
    /** Submit an input batch to tracker process queue. */
    virtual bool submitInput(const InputParams &inputParams) = 0;
    /** Wait until a batch's process is done.*/
    virtual CompletionStatus waitForCompletion(InputParams &inputParams) = 0;
    /** Flush the request to send the batch downstream. */
    virtual bool flushReqs() = 0;
};

#endif
