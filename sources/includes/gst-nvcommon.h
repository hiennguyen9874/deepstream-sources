/**
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file
 * <b>NVIDIA GStreamer DeepStream: Common Properties</b>
 *
 * @b Description: This file specifies the NVIDIA DeepStream GStreamer common
 * properties functions, useful for reuse by multiple components
 *
 */
#ifndef __GST_NVCOMMON_H__
#define __GST_NVCOMMON_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __aarch64__
#define CHECK_DEFAULT_MEM(memType1, memType2)                                \
    ((memType1 == NVBUF_MEM_DEFAULT && memType2 == NVBUF_MEM_CUDA_DEVICE) || \
     (memType1 == NVBUF_MEM_CUDA_DEVICE && memType2 == NVBUF_MEM_DEFAULT))
#else
#define CHECK_DEFAULT_MEM(memType1, memType2)                                  \
    ((memType1 == NVBUF_MEM_DEFAULT && memType2 == NVBUF_MEM_SURFACE_ARRAY) || \
     (memType1 == NVBUF_MEM_SURFACE_ARRAY && memType2 == NVBUF_MEM_DEFAULT))
#endif

#define CHECK_NVDS_MEMORY_AND_GPUID(object, surface)                                             \
    ({                                                                                           \
        int _errtype = 0;                                                                        \
        do {                                                                                     \
            if (((surface->memType == NVBUF_MEM_DEFAULT ||                                       \
                  surface->memType == NVBUF_MEM_CUDA_DEVICE) &&                                  \
                 ((int)surface->gpuId != (int)object->gpu_id)) ||                                \
                (((int)surface->gpuId == (int)object->gpu_id) &&                                 \
                 (surface->memType == NVBUF_MEM_SYSTEM))) {                                      \
                GST_ELEMENT_ERROR(                                                               \
                    object, RESOURCE, FAILED,                                                    \
                    ("Memory Compatibility Error:Input surface gpu-id doesnt match with "        \
                     "configured gpu-id for element,"                                            \
                     " please allocate input using unified memory, or use same gpu-ids OR,"      \
                     " if same gpu-ids are used ensure appropriate Cuda memories are used"),     \
                    ("surface-gpu-id=%d,%s-gpu-id=%d", surface->gpuId, GST_ELEMENT_NAME(object), \
                     object->gpu_id));                                                           \
                _errtype = 1;                                                                    \
            }                                                                                    \
        } while (0);                                                                             \
        _errtype;                                                                                \
    })

#define GST_TYPE_NVBUF_MEMORY_TYPE (gst_nvbuf_memory_get_type())
GType gst_nvbuf_memory_get_type(void);

#define GST_TYPE_COMPUTE_HW (gst_compute_hw_get_type())
GType gst_compute_hw_get_type(void);

#define GST_TYPE_INTERPOLATION_METHOD (gst_video_interpolation_method_get_type())
GType gst_video_interpolation_method_get_type(void);

#define PROP_NVDS_GPU_ID_INSTALL(gobject_class)                                          \
    do {                                                                                 \
        g_object_class_install_property(                                                 \
            gobject_class, PROP_GPU_DEVICE_ID,                                           \
            g_param_spec_uint("gpu-id", "Set GPU Device ID for operation",               \
                              "Set GPU Device ID for operation", 0, G_MAXUINT, 0,        \
                              (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | \
                                            GST_PARAM_MUTABLE_READY)));                  \
    } while (0)

#define PROP_NVBUF_MEMORY_TYPE_INSTALL(gobject_class)                                           \
    do {                                                                                        \
        g_object_class_install_property(                                                        \
            gobject_class, PROP_NVBUF_MEMORY_TYPE,                                              \
            g_param_spec_enum("nvbuf-memory-type", "Type of NvBufSurface memory allocated",     \
                              "Type of NvBufSurface Memory to be allocated for output buffers", \
                              GST_TYPE_NVBUF_MEMORY_TYPE, NVBUF_MEM_DEFAULT,                    \
                              (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |        \
                                            GST_PARAM_MUTABLE_READY)));                         \
    } while (0)

#define PROP_COMPUTE_HW_INSTALL(gobject_class)                                             \
    do {                                                                                   \
        g_object_class_install_property(                                                   \
            gobject_class, PROP_COMPUTE_HW,                                                \
            g_param_spec_enum("compute-hw", "compute-hw", "Compute Scaling HW",            \
                              GST_TYPE_COMPUTE_HW, NvBufSurfTransformCompute_Default,      \
                              (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |   \
                                            GST_PARAM_CONTROLLABLE | G_PARAM_CONSTRUCT))); \
    } while (0)

#define PROP_INTERPOLATION_METHOD_INSTALL(gobject_class)                                   \
    do {                                                                                   \
        g_object_class_install_property(                                                   \
            gobject_class, PROP_INTERPOLATION_METHOD,                                      \
            g_param_spec_enum("interpolation-method", "Interpolation-method",              \
                              "Set interpolation methods", GST_TYPE_INTERPOLATION_METHOD,  \
                              NvBufSurfTransformInter_Default,                             \
                              (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |   \
                                            GST_PARAM_CONTROLLABLE | G_PARAM_CONSTRUCT))); \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif
