#ifndef _GST_NV_VIDEO_TEST_SRC_H_
#define _GST_NV_VIDEO_TEST_SRC_H_

#include <gst/base/base.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <nvbufsurface.h>

G_BEGIN_DECLS

#define GST_TYPE_NV_VIDEO_TEST_SRC (gst_nv_video_test_src_get_type())
#define GST_NV_VIDEO_TEST_SRC(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NV_VIDEO_TEST_SRC, GstNvVideoTestSrc))
#define GST_NV_VIDEO_TEST_SRC_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NV_VIDEO_TEST_SRC, GstNvVideoTestSrcClass))
#define GST_IS_NV_VIDEO_TEST_SRC(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NV_VIDEO_TEST_SRC))
#define GST_IS_NV_VIDEO_TEST_SRC_CLASS(obj) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NV_VIDEO_TEST_SRC))

typedef enum {
    GST_NV_VIDEO_TEST_SRC_SMPTE,
    GST_NV_VIDEO_TEST_SRC_MANDELBROT,
    GST_NV_VIDEO_TEST_SRC_GRADIENT
} GstNvVideoTestSrcPattern;

typedef enum {
    GST_NV_VIDEO_TEST_SRC_FRAMES,
    GST_NV_VIDEO_TEST_SRC_WALL_TIME,
    GST_NV_VIDEO_TEST_SRC_RUNNING_TIME
} GstNvVideoTestSrcAnimationMode;

typedef struct _GstNvVideoTestSrc GstNvVideoTestSrc;
typedef struct _GstNvVideoTestSrcClass GstNvVideoTestSrcClass;

struct _GstNvVideoTestSrc {
    GstPushSrc parent;

    // Plugin parameters.
    GstNvVideoTestSrcPattern pattern;
    GstNvVideoTestSrcAnimationMode animation_mode;
    guint gpu_id;
    NvBufSurfaceMemType memtype;
    gboolean enable_rdma;

    // Stream details set during caps negotiation.
    GstCaps *caps;
    GstVideoInfo info;

    // Runtime state.
    GstClockTime running_time;
    guint filled_frames;

    NvBufSurfaceParams *cuda_surf;
    unsigned int cuda_block_size;
    unsigned int cuda_num_blocks;
    void (*cuda_fill_image)(GstNvVideoTestSrc *src);
};

struct _GstNvVideoTestSrcClass {
    GstPushSrcClass parent_class;
};

GType gst_nv_video_test_src_get_type(void);

G_END_DECLS

#endif
