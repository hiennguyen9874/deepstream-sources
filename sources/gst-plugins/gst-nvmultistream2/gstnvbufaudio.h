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

/**
 * @file gstnvbufaudio.h
 * <b>Gst Helper APIs for NvBufAudio Interface </b>
 *
 * This file specifies the API to fetch audio specific
 * information from GstCaps.
 *
 */

#ifndef _GST_NVSTREAMMUX_AUDIO_H_
#define _GST_NVSTREAMMUX_AUDIO_H_

#include <gst/audio/audio.h>
#include <gst/gst.h>

#include "gstnvstreammuxdebug.h"
#include "nvbufaudio.h"

class GstNvBufAudioCaps {
public:
    GstNvBufAudioCaps(GstCaps *aCaps);
    NvBufAudioFormat GetAudioFormat();
    NvBufAudioLayout GetAudioLayout();
    uint32_t GetAudioRate();
    uint32_t GetAudioChannels();
    bool GetAudioParams(NvBufAudioParams &aAudioParams);

private:
    gchar const *GetFieldStringValue(gchar const *fieldName);
    uint32_t const GetFieldIntValue(gchar const *fieldName);

    GstCaps *caps;
    GstStructure *capsStruct;
    GstAudioInfo audioInfo;
    NvBufAudioParams audioParams;

    /** audio/x-raw
     *   format: { S8, U8, S16LE, S16BE, U16LE, U16BE, S24_32LE, S24_32BE,
     *             U24_32LE, U24_32BE, S32LE, S32BE, U32LE, U32BE, S24LE,
     *             S24BE, U24LE, U24BE, S20LE, S20BE, U20LE, U20BE, S18LE,
     *             S18BE, U18LE, U18BE, F32LE, F32BE, F64LE, F64BE }
     *   rate: [ 1, 2147483647 ]
     *   channels: [ 1, 2147483647 ]
     *   layout: interleaved
     */
    gchar const *format;
    gchar const *rate;
    gchar const *channels;
    gchar const *layout;
};

#endif /**< _GST_NVSTREAMMUX_AUDIO_H_ */
