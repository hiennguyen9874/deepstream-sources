#ifndef __GST_NVDSCUSTOMUSER_META_H__
#define __GST_NVDSCUSTOMUSER_META_H__

#include <nvdsmeta.h>

#define NVDS_USER_CUSTOM_META (nvds_get_user_meta_type((gchar *)"NVIDIA.USER.CUSTOM_META"))

typedef struct _NVDS_CUSTOM_PAYLOAD {
    uint32_t payloadType;
    uint32_t payloadSize;
    uint8_t *payload;
} NVDS_CUSTOM_PAYLOAD;

#endif //__GST_NVDSCUSTOMUSER_META_H__
