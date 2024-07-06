#ifndef __NVGSTDS_PRIMARY_GIE_H__
#define __NVGSTDS_PRIMARY_GIE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "deepstream_gie.h"

typedef struct {
    GstElement *bin;
    GstElement *queue;
    GstElement *nvvidconv;
    GstElement *primary_gie;
} NvDsPrimaryGieBin;

/**
 * Initialize @ref NvDsPrimaryGieBin. It creates and adds primary infer and
 * other elements needed for processing to the bin.
 * It also sets properties mentioned in the configuration file under
 * group @ref CONFIG_GROUP_PRIMARY_GIE
 *
 * @param[in] config pointer to infer @ref NvDsGieConfig parsed from
 *            configuration file.
 * @param[in] bin pointer to @ref NvDsPrimaryGieBin to be filled.
 *
 * @return true if bin created successfully.
 */
gboolean create_primary_gie_bin(NvDsGieConfig *config, NvDsPrimaryGieBin *bin);

#ifdef __cplusplus
}
#endif

#endif
