/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gstnvdsbufferpool.h>

#include "gstnvvideotestsrc.h"
#include "patterns.h"

GST_DEBUG_CATEGORY_STATIC(nv_video_test_src_debug);
#define GST_CAT_DEFAULT nv_video_test_src_debug

#define DEFAULT_WIDTH (1280)
#define DEFAULT_HEIGHT (720)
#define DEFAULT_FRAMERATE (60)
#define DEFAULT_PATTERN (GST_NV_VIDEO_TEST_SRC_SMPTE)
#define DEFAULT_ANIMATION_MODE (GST_NV_VIDEO_TEST_SRC_FRAMES)
#define DEFAULT_GPU_ID (0)
#define DEFAULT_MEMTYPE (NVBUF_MEM_CUDA_DEVICE)

enum {
    PROP_0,
    PROP_PATTERN,
    PROP_ANIMATION_MODE,
    PROP_GPU_ID,
    PROP_MEMTYPE,
};

static GstStaticPadTemplate gst_nv_video_test_src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES("memory:NVMM", "{I420, NV12, RGBA}")));

#define parent_class gst_nv_video_test_src_parent_class
G_DEFINE_TYPE(GstNvVideoTestSrc, gst_nv_video_test_src, GST_TYPE_PUSH_SRC)

static void gst_nv_video_test_src_set_pattern(GstNvVideoTestSrc *self, int pattern);
static void gst_nv_video_test_src_set_property(GObject *object,
                                               guint property_id,
                                               const GValue *value,
                                               GParamSpec *pspec);
static void gst_nv_video_test_src_get_property(GObject *object,
                                               guint property_id,
                                               GValue *value,
                                               GParamSpec *pspec);

static gboolean gst_nv_video_test_src_set_caps(GstBaseSrc *bsrc, GstCaps *caps);
static GstCaps *gst_nv_video_test_src_fixate(GstBaseSrc *bsrc, GstCaps *caps);
static gboolean gst_nv_video_test_src_decide_allocation(GstBaseSrc *bsrc, GstQuery *query);
static gboolean gst_nv_video_test_src_start(GstBaseSrc *self);
static gboolean gst_nv_video_test_src_stop(GstBaseSrc *self);

static GstFlowReturn gst_nv_video_test_src_fill(GstPushSrc *psrc, GstBuffer *buffer);

#define GST_TYPE_NV_VIDEO_TEST_SRC_PATTERN (gst_nv_video_test_src_pattern_get_type())
static GType gst_nv_video_test_src_pattern_get_type(void)
{
    static gsize id = 0;
    static const GEnumValue patterns[] = {
        {GST_NV_VIDEO_TEST_SRC_SMPTE, "SMPTE color bars", "smpte"},
        {GST_NV_VIDEO_TEST_SRC_MANDELBROT, "Mandelbrot set", "mandelbrot"},
        {GST_NV_VIDEO_TEST_SRC_GRADIENT, "Gradient", "gradient"},
        {0, NULL, NULL}};

    if (g_once_init_enter(&id)) {
        GType tmp = g_enum_register_static("GstNvVideoTestSrcPattern", patterns);
        g_once_init_leave(&id, tmp);
    }

    return (GType)id;
}

#define GST_TYPE_NV_VIDEO_TEST_SRC_ANIMATION_MODE (gst_nv_video_test_src_animation_mode_get_type())
static GType gst_nv_video_test_src_animation_mode_get_type(void)
{
    static gsize id = 0;
    static const GEnumValue modes[] = {
        {GST_NV_VIDEO_TEST_SRC_FRAMES, "Frame count", "frames"},
        {GST_NV_VIDEO_TEST_SRC_WALL_TIME, "Wall clock time", "wall-time"},
        {GST_NV_VIDEO_TEST_SRC_RUNNING_TIME, "Running time", "running-time"},
        {0, NULL, NULL}};

    if (g_once_init_enter(&id)) {
        GType tmp = g_enum_register_static("GstNvVideoTestSrcAnimationMode", modes);
        g_once_init_leave(&id, tmp);
    }

    return (GType)id;
}

#define GST_TYPE_NVMM_MEMTYPE (gst_nvmm_memtype_get_type())
static GType gst_nvmm_memtype_get_type(void)
{
    static gsize id = 0;
    static const GEnumValue modes[] = {{NVBUF_MEM_CUDA_PINNED, "host", "Host/Pinned CUDA memory"},
                                       {NVBUF_MEM_CUDA_DEVICE, "device", "Device CUDA memory"},
                                       {NVBUF_MEM_CUDA_UNIFIED, "unified", "Unified CUDA memory"},
                                       {0, NULL, NULL}};

    if (g_once_init_enter(&id)) {
        GType tmp = g_enum_register_static("NvBufSurfaceMemType", modes);
        g_once_init_leave(&id, tmp);
    }

    return (GType)id;
}

static void gst_nv_video_test_src_class_init(GstNvVideoTestSrcClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GstBaseSrcClass *basesrc_class = GST_BASE_SRC_CLASS(klass);
    GstPushSrcClass *pushsrc_class = GST_PUSH_SRC_CLASS(klass);

    gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_get_property);

    g_object_class_install_property(
        gobject_class, PROP_PATTERN,
        g_param_spec_enum("pattern", "Pattern", "Type of test pattern to generate",
                          GST_TYPE_NV_VIDEO_TEST_SRC_PATTERN, DEFAULT_PATTERN,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_ANIMATION_MODE,
        g_param_spec_enum("animation-mode", "Animation mode",
                          "For animating patterns, the counter that controls the animation.",
                          GST_TYPE_NV_VIDEO_TEST_SRC_ANIMATION_MODE, DEFAULT_ANIMATION_MODE,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_GPU_ID,
        g_param_spec_uint("gpu-id", "GPU ID", "ID of the GPU where the buffers are allocated", 0,
                          G_MAXINT, DEFAULT_GPU_ID,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_MEMTYPE,
        g_param_spec_enum("memtype", "Memory type",
                          "Type of the memory used for the allocated buffers",
                          GST_TYPE_NVMM_MEMTYPE, DEFAULT_MEMTYPE,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_set_static_metadata(element_class, GST_PACKAGE_NAME, "Source/Video",
                                          GST_PACKAGE_DESCRIPTION, GST_PACKAGE_AUTHOR);

    gst_element_class_add_pad_template(
        element_class, gst_static_pad_template_get(&gst_nv_video_test_src_template));

    basesrc_class->set_caps = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_set_caps);
    basesrc_class->fixate = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_fixate);
    basesrc_class->decide_allocation = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_decide_allocation);
    basesrc_class->start = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_start);
    basesrc_class->stop = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_stop);

    pushsrc_class->fill = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_fill);
}

static void gst_nv_video_test_src_init(GstNvVideoTestSrc *self)
{
    gst_nv_video_test_src_set_pattern(self, DEFAULT_PATTERN);

    self->animation_mode = DEFAULT_ANIMATION_MODE;
    self->gpu_id = DEFAULT_GPU_ID;
    self->memtype = DEFAULT_MEMTYPE;

    gst_base_src_set_format(GST_BASE_SRC_CAST(self), GST_FORMAT_TIME);
    gst_base_src_set_live(GST_BASE_SRC_CAST(self), TRUE);
}

static void gst_nv_video_test_src_set_pattern(GstNvVideoTestSrc *self, int pattern)
{
    self->pattern = (GstNvVideoTestSrcPattern)pattern;

    switch (pattern) {
    case GST_NV_VIDEO_TEST_SRC_SMPTE:
        self->cuda_fill_image = gst_nv_video_test_src_smpte;
        break;

    case GST_NV_VIDEO_TEST_SRC_MANDELBROT:
        self->cuda_fill_image = gst_nv_video_test_src_mandelbrot;
        break;

    case GST_NV_VIDEO_TEST_SRC_GRADIENT:
        self->cuda_fill_image = gst_nv_video_test_src_gradient;
        break;

    default:
        g_assert_not_reached();
    }
}

static void gst_nv_video_test_src_set_property(GObject *object,
                                               guint property_id,
                                               const GValue *value,
                                               GParamSpec *pspec)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(object);

    switch (property_id) {
    case PROP_PATTERN:
        gst_nv_video_test_src_set_pattern(self, g_value_get_enum(value));
        break;

    case PROP_ANIMATION_MODE:
        self->animation_mode = (GstNvVideoTestSrcAnimationMode)g_value_get_enum(value);
        break;

    case PROP_GPU_ID:
        self->gpu_id = g_value_get_uint(value);
        break;

    case PROP_MEMTYPE:
        self->memtype = (NvBufSurfaceMemType)g_value_get_enum(value);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
        break;
    }
}

static void gst_nv_video_test_src_get_property(GObject *object,
                                               guint property_id,
                                               GValue *value,
                                               GParamSpec *pspec)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(object);

    switch (property_id) {
    case PROP_PATTERN:
        g_value_set_enum(value, self->pattern);
        break;

    case PROP_ANIMATION_MODE:
        g_value_set_enum(value, self->animation_mode);
        break;

    case PROP_GPU_ID:
        g_value_set_uint(value, self->gpu_id);
        break;

    case PROP_MEMTYPE:
        g_value_set_enum(value, self->memtype);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
        break;
    }
}

static gboolean gst_nv_video_test_src_set_caps(GstBaseSrc *bsrc, GstCaps *caps)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(bsrc);

    GST_OBJECT_LOCK(self);

    if (!gst_video_info_from_caps(&self->info, caps))
        return FALSE;

    self->caps = caps;

    gst_nv_video_test_src_cuda_init(self);

    GST_OBJECT_UNLOCK(self);

    return TRUE;
}

static GstCaps *gst_nv_video_test_src_fixate(GstBaseSrc *bsrc, GstCaps *caps)
{
    GstStructure *structure;

    caps = gst_caps_make_writable(caps);
    structure = gst_caps_get_structure(caps, 0);

    gst_structure_fixate_field_nearest_int(structure, "width", DEFAULT_WIDTH);
    gst_structure_fixate_field_nearest_int(structure, "height", DEFAULT_HEIGHT);

    if (gst_structure_has_field(structure, "framerate")) {
        gst_structure_fixate_field_nearest_fraction(structure, "framerate", DEFAULT_FRAMERATE, 1);
    } else {
        gst_structure_set(structure, "framerate", GST_TYPE_FRACTION, DEFAULT_FRAMERATE, 1, NULL);
    }

    caps = GST_BASE_SRC_CLASS(parent_class)->fixate(bsrc, caps);

    return caps;
}

static gboolean gst_nv_video_test_src_decide_allocation(GstBaseSrc *bsrc, GstQuery *query)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(bsrc);

    // Remove any downstream-proposed allocation pools and params, since these
    // may not meet the requirements of our pool (e.g. memory type and GPU ID).
    while (gst_query_get_n_allocation_pools(query) > 0)
        gst_query_remove_nth_allocation_pool(query, 0);
    while (gst_query_get_n_allocation_params(query) > 0)
        gst_query_remove_nth_allocation_param(query, 0);

    // Allocate and configure a new GstNvDsBufferPool.
    GstBufferPool *pool = gst_nvds_buffer_pool_new();
    GstStructure *config = gst_buffer_pool_get_config(pool);

    // Note that the size of each GstBuffer allocated by the GstNvmmBufferPool
    // is equal to the size of the NvBufSurface struct. The GstBuffer memory
    // simply contains the NvBufSurface descriptor which in turn points to the
    // actual NVMM GPU buffer allocation(s).
    gst_buffer_pool_config_set_params(config, self->caps, sizeof(NvBufSurface), 0, 0);

    // Configure the GstNvDsBufferPool allocator.
    gst_structure_set(config, "memtype", G_TYPE_UINT, self->memtype, "gpu-id", G_TYPE_UINT,
                      self->gpu_id, NULL);

    if (!gst_buffer_pool_set_config(pool, config)) {
        GST_ERROR("Failed to set buffer pool config");
        return FALSE;
    }

    gst_query_add_allocation_pool(query, pool, sizeof(NvBufSurface), 0, 0);

    gst_object_unref(pool);

    return GST_BASE_SRC_CLASS(parent_class)->decide_allocation(bsrc, query);
}

static gboolean gst_nv_video_test_src_start(GstBaseSrc *bsrc)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(bsrc);

    GST_OBJECT_LOCK(self);

    gst_video_info_init(&self->info);

    self->running_time = 0;
    self->filled_frames = 0;

    GST_OBJECT_UNLOCK(self);

    return TRUE;
}

static gboolean gst_nv_video_test_src_stop(GstBaseSrc *bsrc)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(bsrc);

    GST_OBJECT_LOCK(self);

    gst_nv_video_test_src_cuda_free(self);

    GST_OBJECT_UNLOCK(self);

    return TRUE;
}

static GstFlowReturn gst_nv_video_test_src_fill(GstPushSrc *psrc, GstBuffer *buffer)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(psrc);
    GstMapInfo map;

    // If 0 framerate and we've returned a frame, EOS.
    if (G_UNLIKELY(self->info.fps_n == 0 && self->filled_frames == 1))
        return GST_FLOW_EOS;

    gst_buffer_map(buffer, &map, GST_MAP_READWRITE);

    // The memory of a GstBuffer allocated by an NvDsBufferPool contains
    // the NvBufSurface descriptor which then describes the actual GPU
    // buffer allocation(s) in its surfaceList.
    NvBufSurface *surf = (NvBufSurface *)map.data;

    // Use CUDA to fill the GPU buffer with a test pattern.
    //
    // NOTE: In this source, we currently only fill the GPU buffer using CUDA.
    //       This source could be modified to fill the buffer instead with other
    //       mechanisms such as mapped CPU writes or RDMA transfers:
    //
    //       1) To use mapped CPU writes, the GPU buffer could be mapped into
    //          the CPU address space using NvBufSurfaceMap.
    //
    //       2) To use RDMA transfers from another hardware device, the GPU
    //          address for the buffer(s) (i.e. the `dataPtr` member of the
    //          NvBufSurfaceParams) could be passed here to the device driver
    //          that is responsible for performing the RDMA transfer into the
    //          buffer. Details on how RDMA to NVIDIA GPUs can be performed by
    //          device drivers is provided in a demonstration application
    //          available at https://github.com/NVIDIA/jetson-rdma-picoevb.
    //
    //       For more details on the NvBufSurface API, see nvbufsurface.h
    //
    gst_nv_video_test_src_cuda_prepare(self, surf->surfaceList);
    self->cuda_fill_image(self);

    // Set the numFilled field of the surface with the number of surfaces
    // that have been filled (always 1 in this example).
    // This metadata is required by other downstream DeepStream plugins.
    surf->numFilled = 1;

    gst_buffer_unmap(buffer, &map);

    // Set the buffer timestamps and duration.
    GST_BUFFER_PTS(buffer) = self->running_time;
    GST_BUFFER_DTS(buffer) = GST_CLOCK_TIME_NONE;
    GST_BUFFER_OFFSET(buffer) = self->filled_frames;
    self->filled_frames++;
    GST_BUFFER_OFFSET_END(buffer) = self->filled_frames;

    if (self->info.fps_n != 0) {
        GstClockTime next_time = gst_util_uint64_scale(
            self->filled_frames, self->info.fps_d * GST_SECOND, self->info.fps_n);
        GST_BUFFER_DURATION(buffer) = next_time - self->running_time;
        self->running_time = next_time;
    } else {
        GST_BUFFER_DURATION(buffer) = GST_CLOCK_TIME_NONE;
    }

    gst_object_sync_values(GST_OBJECT(psrc), GST_BUFFER_PTS(buffer));

    return GST_FLOW_OK;
}

static gboolean plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(nv_video_test_src_debug, "nvvideotestsrc", 0, GST_PACKAGE_NAME);

    return gst_element_register(plugin, "nvvideotestsrc", GST_RANK_PRIMARY,
                                GST_TYPE_NV_VIDEO_TEST_SRC);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  nvdsgst_videotestsrc,
                  GST_PACKAGE_DESCRIPTION,
                  plugin_init,
                  GST_PACKAGE_VERSION,
                  GST_PACKAGE_LICENSE,
                  GST_PACKAGE_NAME,
                  GST_PACKAGE_ORIGIN)
