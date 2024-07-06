#ifndef _GST_NVSTREAMMUX_DEBUG_H_
#define _GST_NVSTREAMMUX_DEBUG_H_

#include <gst/gstinfo.h>
#include <stdarg.h>
#include <stdio.h>

#include "nvstreammux_debug.h"

#if 1
#define LOGD(...)
#else
#define LOGD(fmt, ...) printf("[DEBUG %s %d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
#endif

#define LOGV(fmt, ...) printf("[VERBOSE %s %d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
#define LOGE(fmt, ...) printf("[ERROR %s %d] " fmt, __func__, __LINE__, ##__VA_ARGS__)

#endif /**< _GST_NVSTREAMMUX_DEBUG_H_ */
