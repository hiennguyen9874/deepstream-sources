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

#ifndef __GSTNVSTREAMMUX_NTP__
#define __GSTNVSTREAMMUX_NTP__

#include <glib.h>
#include <gst/gst.h>

G_BEGIN_DECLS

#define NVDS_RFC3339_STR_BUF_LEN 32

void generate_rfc3339_str_from_ts(gchar *buf, GstClockTime ts);

typedef enum {
    GST_NVDS_NTP_CALC_MODE_SYSTEM_TIME,
    GST_NVDS_NTP_CALC_MODE_RTCP
} GstNvDsNtpCalculatorMode;

/* Modes of NTP timestamp correction based on frame rate */
typedef enum {
    /* Frame rate based correction disabled */
    GST_NVDS_NTP_CORRECTION_DISABLED,
    /* Use the average frame rate provided in the NTP query to correct
     * NTP timestamp. (PTS based frame rate calculation at rtpjitterbuffer.) */
    GST_NVDS_NTP_CORRECTION_AUTOMATIC,
    /* Use the frame rate provided by application to correct NTP timestamp.
     * This mode is required frame rate is different at rtpjitterbuffer and
     * nvstreammux.  Bug 3620472, 3626628  */
    GST_NVDS_NTP_CORRECTION_USER_FRAME_RATE
} GstNvDsNtpCorrectionMode;

typedef struct _GstNvDsNtpCalculator GstNvDsNtpCalculator;

GstNvDsNtpCalculator *gst_nvds_ntp_calculator_new(GstNvDsNtpCalculatorMode mode,
                                                  GstClockTime frame_duration,
                                                  GstElement *elem,
                                                  guint source_id);

void gst_nvds_ntp_calculator_add_ntp_sync_values(GstNvDsNtpCalculator *calc,
                                                 GstClockTime ntp_time_epoch_ns,
                                                 GstClockTime ntp_frame_timestamp,
                                                 GstClockTime avg_frame_time);

gboolean gst_nvds_ntp_calculator_have_ntp_sync_values(GstNvDsNtpCalculator *calc);

void gst_nvds_ntp_calculator_get_ntp_sync_values(GstNvDsNtpCalculator *calc,
                                                 GstClockTime *ntp_time_epoch_ns,
                                                 GstClockTime *ntp_frame_timestamp,
                                                 GstClockTime *avg_frame_time,
                                                 GstClockTime *ntp_time_epoch_ns_next,
                                                 GstClockTime *ntp_frame_timestamp_next);

GstClockTime gst_nvds_ntp_calculator_get_buffer_ntp(GstNvDsNtpCalculator *calc,
                                                    GstClockTime buf_pts);

void gst_nvds_ntp_calculator_reset(GstNvDsNtpCalculator *calc);

void gst_nvds_ntp_calculator_free(GstNvDsNtpCalculator *calc);

G_END_DECLS

#endif
