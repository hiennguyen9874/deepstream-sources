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

#ifndef __GST_NVSTREAMPAD_H__
#define __GST_NVSTREAMPAD_H__

#include <gst/gst.h>

GType gst_nvstream_pad_get_type(void);
#define GST_TYPE_NVSTREAM_PAD (gst_nvstream_pad_get_type())
#define GST_NVSTREAM_PAD(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVSTREAM_PAD, GstNvStreamPad))
#define GST_NVSTREAM_PAD_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVSTREAM_PAD, GstNvStreamPadClass))
#define GST_IS_NVSTREAM_PAD(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVSTREAM_PAD))
#define GST_IS_NVSTREAM_PAD_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVSTREAM_PAD))
#define GST_NVSTREAM_PAD_CAST(obj) ((GstNvStreamPad *)(obj))

typedef struct _GstNvStreamPad GstNvStreamPad;
typedef struct _GstNvStreamPadClass GstNvStreamPadClass;

struct _GstNvStreamPad {
    GstPad parent;

    gboolean got_eos;
};

struct _GstNvStreamPadClass {
    GstPadClass parent;
};

#endif
