#ifndef _DS_YAML_PARSER_H_
#define _DS_YAML_PARSER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>

#include "nvds_yml_parser.h"

NvDsYamlParserStatus ds_parse_rtsp_output(GstElement *sink,
                                          GstRTSPServer *server,
                                          GstRTSPMediaFactory *factory,
                                          gchar *cfg_file_path,
                                          const char *group);

NvDsYamlParserStatus ds_parse_enc_config(GstElement *encoder,
                                         gchar *cfg_file_path,
                                         const char *group);

guint ds_parse_group_type(gchar *cfg_file_path, const char *group);

guint ds_parse_enc_type(gchar *cfg_file_path, const char *group);

GString *ds_parse_file_name(gchar *cfg_file_path, const char *group);

GString *ds_parse_config_yml_filepath(gchar *cfg_file_path, const char *group);

NvDsYamlParserStatus ds_parse_videotemplate_config(GstElement *vtemplate,
                                                   gchar *cfg_file_path,
                                                   const char *group);

#ifdef __cplusplus
}
#endif

#endif
