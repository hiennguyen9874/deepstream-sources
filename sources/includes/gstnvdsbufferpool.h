#ifndef GSTNVDSBUFFERPOOL_H_
#define GSTNVDSBUFFERPOOL_H_

#include <gst/gst.h>

G_BEGIN_DECLS

typedef struct _GstNvDsBufferPool GstNvDsBufferPool;
typedef struct _GstNvDsBufferPoolClass GstNvDsBufferPoolClass;
typedef struct _GstNvDsBufferPoolPrivate GstNvDsBufferPoolPrivate;

#define GST_TYPE_NVDS_BUFFER_POOL (gst_nvds_buffer_pool_get_type())
#define GST_IS_NVDS_BUFFER_POOL(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVDS_BUFFER_POOL))
#define GST_NVDS_BUFFER_POOL(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVDS_BUFFER_POOL, GstNvDsBufferPool))
#define GST_NVDS_BUFFER_POOL_CAST(obj) ((GstNvDsBufferPool *)(obj))

#define GST_NVDS_MEMORY_TYPE "nvds"
#define GST_BUFFER_POOL_OPTION_NVDS_META "GstBufferPoolOptionNvDsMeta"

struct _GstNvDsBufferPool {
    GstBufferPool bufferpool;

    GstNvDsBufferPoolPrivate *priv;
};

struct _GstNvDsBufferPoolClass {
    GstBufferPoolClass parent_class;
};

GType gst_nvds_buffer_pool_get_type(void);

GstBufferPool *gst_nvds_buffer_pool_new(void);

G_END_DECLS

#endif /* GSTNVDSBUFFERPOOL_H_ */
