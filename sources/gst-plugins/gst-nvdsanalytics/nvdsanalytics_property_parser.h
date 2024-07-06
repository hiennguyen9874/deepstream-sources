#ifndef NVDSANALYTICS_PROPERTY_FILE_PARSER_H_
#define NVDSANALYTICS_PROPERTY_FILE_PARSER_H_

#include <gst/gst.h>

#include "gstnvdsanalytics.h"

gboolean nvdsanalytics_parse_config_file(GstNvDsAnalytics *nvdsanalytics, gchar *cfg_file_path);

#endif /* NVDSANALYTICS_PROPERTY_FILE_PARSER_H_ */
