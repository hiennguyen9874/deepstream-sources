#ifndef __NVGSTDS_AUDIO_CLASSIFIER_H__
#define __NVGSTDS_AUDIO_CLASSIFIER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "deepstream_gie.h"

typedef struct {
    GstElement *bin;
    GstElement *queue;
    GstElement *classifier;
} NvDsAudioClassifierBin;

/**
 * Initialize @ref NvDsAudioClassifierBin. It creates and adds primary infer and
 * other elements needed for processing to the bin.
 * It also sets properties mentioned in the configuration file under
 * group @ref CONFIG_GROUP_AUDIO_CLASSIFIER
 *
 * @param[in] config pointer to infer @ref NvDsGieConfig parsed from
 *            configuration file.
 * @param[in] bin pointer to @ref NvDsAudioClassifierBin to be filled.
 *
 * @return true if bin created successfully.
 */
gboolean create_audio_classifier_bin(NvDsGieConfig *config, NvDsAudioClassifierBin *bin);

#ifdef __cplusplus
}
#endif

#endif
