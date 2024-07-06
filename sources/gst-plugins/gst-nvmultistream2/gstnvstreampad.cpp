#include "gstnvstreampad.h"

G_DEFINE_TYPE(GstNvStreamPad, gst_nvstream_pad, GST_TYPE_PAD);

static void gst_nvstream_pad_class_init(GstNvStreamPadClass *klass)
{
}

static void gst_nvstream_pad_init(GstNvStreamPad *pad)
{
    pad->got_eos = FALSE;
}
