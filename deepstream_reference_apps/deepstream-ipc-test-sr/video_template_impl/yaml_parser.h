#ifndef __YAML_PARSER_H__
#define __YAML_PARSER_H__
#include <glib.h>
#include <nvdsinfer_context.h>

/*
 * parameters of calibrator
 */
struct cfg_params {
    /* model tensor width*/
    int m_tensor_width;
    /*sr model tensor height*/
    int m_tensor_height;
};

gboolean gst_parse_context_params_yaml(const gchar *cfg_file_path, cfg_params &cal_params);

#endif