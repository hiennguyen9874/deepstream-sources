#ifndef __NVGSTDS_C2D_MSG_H__
#define __NVGSTDS_C2D_MSG_H__

#include <gst/gst.h>

#include "nvmsgbroker.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NvDsC2DContext {
    gpointer libHandle;
    gchar *protoLib;
    gchar *connStr;
    gchar *configFile;
    gpointer uData;
    GHashTable *hashMap;
    NvMsgBrokerClientHandle connHandle;
    nv_msgbroker_subscribe_cb_t subscribeCb;
} NvDsC2DContext;

typedef struct NvDsMsgConsumerConfig {
    gboolean enable;
    gchar *proto_lib;
    gchar *conn_str;
    gchar *config_file_path;
    GPtrArray *topicList;
    gchar *sensor_list_file;
} NvDsMsgConsumerConfig;

NvDsC2DContext *start_cloud_to_device_messaging(NvDsMsgConsumerConfig *config,
                                                nv_msgbroker_subscribe_cb_t cb,
                                                void *uData);
gboolean stop_cloud_to_device_messaging(NvDsC2DContext *uCtx);

#ifdef __cplusplus
}
#endif
#endif
