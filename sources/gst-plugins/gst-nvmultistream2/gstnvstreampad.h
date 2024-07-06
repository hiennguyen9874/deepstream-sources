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
