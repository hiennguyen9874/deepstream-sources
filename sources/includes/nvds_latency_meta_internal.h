#ifndef _NVDSMETA_LATENCY_INTERNAL_H_
#define _NVDSMETA_LATENCY_INTERNAL_H_

#include "glib.h"

#ifdef __cplusplus
extern "C" {
#endif

void *nvds_set_latency_metadata_ptr(void);

gpointer nvds_copy_latency_meta(gpointer data, gpointer user_data);

void nvds_release_latency_meta(gpointer data, gpointer user_data);

gdouble nvds_get_current_system_timestamp(void);

gboolean nvds_get_enable_per_component_latency_measurement(void);

#define nvds_enable_component_latency_measurement \
    (nvds_get_enable_per_component_latency_measurement())

#ifdef __cplusplus
}
#endif
#endif
