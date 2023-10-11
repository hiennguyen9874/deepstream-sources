/**
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */
#ifndef _MISCDATABUFMANAGER_H
#define _MISCDATABUFMANAGER_H

#include <gst/gst.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

#include "nvdstracker.h"

/** Tracker misc data buffer for a batch. */
struct NvTrackerMiscDataBuffer {
    /** Past frame objects. */
    NvDsPastFrameObjBatch pastFrameObjBatch;
    /** Target trajectories. */
    NvDsTrajectoryBatch trajectoryBatch;
    /** ReID tensor. */
    NvDsReidTensorBatch reidTensorBatch;
};

/** Tracker misc data memory pool. */
class TrackerMiscDataManager {
public:
    TrackerMiscDataManager();
    ~TrackerMiscDataManager();
    /** Initialize buffer pool based on tracker size information. */
    bool init(uint32_t batchSize,
              uint32_t gpuId,
              uint32_t maxTargetsPerStream,
              uint32_t maxShadowTrackingAge,
              uint32_t reidFeatureSize,
              uint32_t maxBufferPoolSize,
              bool pastFrame,
              bool outputReidTensor);
    /** Return buffer to pool. */
    void returnBuffer(NvTrackerMiscDataBuffer *data);
    /** Pop a buffer from pool. */
    NvTrackerMiscDataBuffer *pop();
    /** Release buffer pool. */
    void deInit();

private:
    /** Buffer pool is empty when there is no misc data.*/
    bool m_IntentionallyEmpty;
    bool m_PastFrame;
    bool m_OutputReidTensor;
    /** Lock to write the free queue. */
    std::mutex m_Mutex;
    std::condition_variable m_Cond;
    /** Unused buffer queue. */
    std::queue<NvTrackerMiscDataBuffer *> m_FreeQueue;
    /** All buffers in the pool. */
    std::vector<NvTrackerMiscDataBuffer *> m_BufferSet;
    /** Getter */
    bool isQueueFull() { return m_BufferSet.size() == m_FreeQueue.size(); }
    /** Allocate memory for past frame objects. */
    void allocatePastFrame(NvTrackerMiscDataBuffer *pNewBuf,
                           uint32_t batchSize,
                           uint32_t maxTargetsPerStream,
                           uint32_t maxShadowTrackingAge);
    /** Release memory for past frame objects. */
    void releasePastFrame(NvTrackerMiscDataBuffer *pBuffer);
    /** Clear objects in memory filled previously. */
    void resetPastFrame(NvTrackerMiscDataBuffer *pBuffer);
    /** Allocate memory for reid tensors. */
    void allocateReid(NvTrackerMiscDataBuffer *pNewBuf,
                      uint32_t batchSize,
                      uint32_t maxTargetsPerStream,
                      uint32_t reidFeatureSize);
    /** Release memory for reid tensors. */
    void releaseReid(NvTrackerMiscDataBuffer *pBuffer);
    /** Clear reid tensors in memory filled previously. */
    void resetReid(NvTrackerMiscDataBuffer *pBuffer);
};

/** GStreamer mini object for GStreamer pipeline to control tracker user meta. */
struct GstNvTrackerMiscDataObject {
    /** GStreamer mini object for refcount. */
    GstMiniObject mini_object;
    /** Pointer to user misc data buffer pool. */
    TrackerMiscDataManager *misc_data_manager;
    /** Pointer to current buffer. */
    NvTrackerMiscDataBuffer *misc_data_buffer;
};

#endif