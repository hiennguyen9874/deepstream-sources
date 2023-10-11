/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights
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

#ifndef __GST_NVSTREAMMUX_PROPERTY_PARSER_H__
#define __GST_NVSTREAMMUX_PROPERTY_PARSER_H__

#include <glib.h>
#include <yaml-cpp/yaml.h>

#include <unordered_map>

#include "nvstreammux_batch.h"

/** @{ Default Streammux Config props */

/** Defaults for PROP_GROUP */
static NvStreammuxBatchMethod constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_BATCH_METHOD_ALGO_TYPE =
    BATCH_METHOD_ROUND_ROBIN;
static guint constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_BATCH_SIZE = 1;
static gboolean constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_ADAPTIVE_BATCHING = TRUE;
static gboolean constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_ENABLE_SOURCE_CONTROL = FALSE;
static gboolean constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_MAX_FPS_CONTROL = FALSE;
static guint constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_FPS_N = 120;
static guint constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_FPS_D = 1;
static guint constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MIN_FPS_N = 5;
static guint constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MIN_FPS_D = 1;
static guint constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_SAME_SOURCE_FRAMES = 1;

/** Defaults for SOURCE_GROUP */
static guint constexpr NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FPS_N = 60;
static guint constexpr NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FPS_D = 1;
static guint constexpr NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MIN_FPS_N = 5;
static guint constexpr NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MIN_FPS_D = 1;
static guint constexpr NVSTREAMMUX_DEFAULT_SOURCE_GROUP_PRIORITY = 0;
static guint constexpr NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FRAMES_PER_BATCH = 1;

/** @} */

/**
 * Data-structures provided by Amiya along with
 * Design-doc:
 * https://docs.google.com/presentation/d/1fgmbUiiSJlUlIWk9ffMvUsW6Nerqg67_Pp-12vywVzs/edit?ts=5c6f53a0#slide=id.g52a9c8f141_0_5
 * and other means
 *  @}
 */

class MuxConfigParser {
public:
    MuxConfigParser();
    ~MuxConfigParser();

    bool SetConfigFile(gchar const *const cfgFilePath);

    /**
     * @brief  Parse the Config file for per-source
     *         properties
     *         Note: For batch-size, if config unavailable in the file,
     *         it shall be set to default only if batchPolicy->batch_size
     *         was not set to a non-zero value by the caller.
     * @param  batchPolicy [IN/OUT] The batchPolicy to
     *         fill the source properties in
     * @return true if successful, false otherwise
     */
    bool ParseConfigs(BatchPolicyConfig *batchPolicy, bool defaults = false, guint numSources = 1);

private:
    void ParseTxtConfigCommonProps(BatchPolicyConfig *batchPolicy, gchar *group, GKeyFile *keyFile);

    bool ParseTxtConfigPerSourceProps(NvStreammuxSourceProps *sourceProps,
                                      gchar *group,
                                      GKeyFile *keyFile);

    bool ParseTxtConfig(BatchPolicyConfig *batchPolicy);

    void ParseYmlConfigCommonProps(BatchPolicyConfig *batchPolicy, std::string group);

    bool ParseYmlConfigPerSourceProps(NvStreammuxSourceProps *sourceProps, std::string group);

    bool ParseYmlConfig(BatchPolicyConfig *batchPolicy);

    gchar *cfgFile;
};

#endif /*__GST_NVSTREAMMUX_PROPERTY_PARSER_H__*/
