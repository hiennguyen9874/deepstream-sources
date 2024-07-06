#ifndef __CONFIG_FILE_PARSER_H_
#define __CONFIG_FILE_PARSER_H_

#include <glib.h>
#include <stdio.h>

typedef struct __NvDsAudioConfig {
    gboolean enable_playback;
    const char *asr_output_file_name;
    gboolean sync;
} NvDsAudioConfig;

typedef struct __NvDsAppConfig {
    gboolean sync;
} NvDsAppConfig;

#endif
