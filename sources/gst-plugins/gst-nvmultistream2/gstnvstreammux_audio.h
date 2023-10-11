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

#ifndef __GST_NVSTREAMMUX_AUDIO_H__
#define __GST_NVSTREAMMUX_AUDIO_H__

#include "gstnvstreammux_impl.h"
#include "nvbufaudio.h"

#define NVSTREAMMUX_BYTES_TO_AUDIOSAMPLESIZE(bytes, channels, bpf) ((bytes) / ((channels) * (bpf)))

#define NVSTREAMMUX_AUDIOSAMPLESIZE_TO_BYTES(samples, channels, bpf) \
    ((samples) * (channels) * (bpf))

extern "C" {
static void audio_mem_buf_unref_callback(gpointer data);
}

/**
 * @brief  The GStreamer wrapper code for NvDsBatchBufferWrapper
 *         which represent one batched buffer.
 *         NOTE: None of the APIs in this class are thread-safe
 */
class GstAudioBatchBufferWrapper : public NvDsBatchBufferWrapper {
public:
    GstAudioBatchBufferWrapper(GstNvStreamMux *mux, unsigned int size, bool is_raw)
        : NvDsBatchBufferWrapper(size), is_raw(is_raw), mux(mux), api((GstElement *)mux)
    {
        batch = malloc(sizeof(NvBufAudio));
        memset(batch, 0, sizeof(NvBufAudio));
        ((NvBufAudio *)batch)->audioBuffers =
            (NvBufAudioParams *)malloc(sizeof(NvBufAudioParams) * size);
        ((NvBufAudio *)batch)->numFilled = 0;
        ((NvBufAudio *)batch)->batchSize = size;
        if (!is_raw) {
            gst_buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY, (NvBufAudio *)batch,
                                                     sizeof(NvBufAudio), 0, sizeof(NvBufAudio),
                                                     (void *)this, audio_mem_buf_unref_callback);
            // gst_buffer_ref(gst_buffer);
        } else {
            raw_batch = gst_buffer_list_new_sized(size);
        }
    }

    void unref() override;
    void reset_batch();
    void dummy() {}
    void copy_meta(unsigned int id,
                   std::shared_ptr<BufferWrapper> src_buffer,
                   unsigned int batch_id,
                   unsigned int frame_number,
                   unsigned int num_surfaces_per_frame,
                   NvDsBatchMeta *dest_batch_meta,
                   unsigned int source_id) override;
    unsigned int copy_buf(std::shared_ptr<BufferWrapper> src, unsigned int pos) override;
    bool push(SourcePad *src_pad, TimePoint current_play_start, NanoSecondsType accum_dur) override;
    bool push(SourcePad *src_pad, unsigned long pts) override;
    void unref_gst_bufs();

    GstBufferList *raw_batch;
    GstBuffer *gst_buffer;
    bool is_raw;
    GstNvStreamMux *mux;
    std::vector<std::shared_ptr<GstBufferWrapper> > gst_in_bufs;

private:
    unsigned int copy_buf_impl(std::shared_ptr<BufferWrapper> buf, unsigned int pos);

private:
    GstCommonBufferAPI api;
};

extern "C" {
static void audio_mem_buf_unref_callback(gpointer data)
{
    if (data != NULL) {
        GstAudioBatchBufferWrapper *batch = (GstAudioBatchBufferWrapper *)data;
        batch->unref_gst_bufs();
        delete batch;
    }
}
}

#endif /**< __GST_NVSTREAMMUX_AUDIO_H__ */
