#ifndef __GST_NVDS_PREPROCESS_YAML_PARSER_H__
#define __GST_NVDS_PREPROCESS_YAML_PARSER_H__

#include <gst/gst.h>

#include "gstnvdspreprocess.h"

/**
 * Parse config file for GstNvDsPreProcess structure.
 *
 * @param nvdspreprocess pointer to GstNvDsPreProcess structure
 *
 * @param cfg_file_path config file path
 *
 * @return boolean denoting if successfully parsed config file
 */
gboolean gst_nvdspreprocess_parse_config_file_yaml(GstNvDsPreProcess *nvdspreprocess,
                                                   const gchar *cfg_file_path);

#endif /* __GST_NVDS_PREPROCESS_YAML_PARSER_H__ */