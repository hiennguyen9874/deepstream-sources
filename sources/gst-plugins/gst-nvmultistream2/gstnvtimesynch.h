/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __GST_NVTIMESYNC_H__
#define __GST_NVTIMESYNC_H__

#include <gst/gst.h>

#include <mutex>
#include <unordered_map>

#include "nvstreammux.h"
#include "nvstreammux_batch.h"
#include "nvstreammux_pads.h"

class NvTimeSync : public ISynchronizeBuffer {
public:
    NvTimeSync(GstElement *el)
        : plugin(el), pipelineLatency(0), upstreamLatency(0), minFpsDuration(0), segments(), mutex()
    {
    }

    BUFFER_TS_STATUS get_synch_info(BufferWrapper *buffer);
    void removing_old_buffer(BufferWrapper *buffer);

    /**
     * @brief  Set the downstream latency
     *         Note: Currently the whole pipelineLatency value is
     *         used in timesynch logic to determine if a buffer is late
     *         at mux input
     *         This include the downstream latency.
     *         Note: This value shall be from the GST_EVENT_LATENCY
     *         sent by the sink plugin.
     *         The mux latency (currently not advertised) is taken care of
     *         by the TimeSynch library (using minFpsDuration)
     * @param  latency [IN] in nanoseconds
     */
    void SetPipelineLatency(GstClockTime latency);

    /**
     * @brief  Set the upstream latency
     * @param  latency [IN] in nanoseconds
     */
    void SetUpstreamLatency(GstClockTime latency);

    GstClockTime GetUpstreamLatency();

    GstClockTime GetCurrentRunningTime();

    void SetSegment(unsigned int stream_id, const GstSegment *segment);

    void SetOperatingMinFpsDuration(NanoSecondsType min_fps_dur);

    NanoSecondsType get_buffer_earlyby_time();

    uint64_t GetBufferRunningTime(uint64_t pts, unsigned int stream_id);

private:
    GstElement *plugin;
    GstClockTime pipelineLatency;
    GstClockTime upstreamLatency;
    GstClockTime minFpsDuration;
    GstClockTime bufferWasEarlyByTime;
    std::unordered_map<unsigned int, GstSegment *> segments;
    std::mutex mutex;
};

#endif
