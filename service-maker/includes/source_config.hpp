/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>SourceConfig definition </b>
 *
 * SourceConfig allows user to configure the source list in a comprehensive
 * way.
 */
#ifndef NVIDIA_DEEPSTREAM_SOURCE_CONFIG
#define NVIDIA_DEEPSTREAM_SOURCE_CONFIG

#include <string>
#include <vector>

#include "object.hpp"

namespace deepstream {

typedef struct {
    std::string uri;
    std::string sensor_id;
    std::string sensor_name;
} SensorInfo;

class SourceConfig {
public:
    /**
     * @brief Create a source config from a yaml config file
     */
    SourceConfig(const std::string &config_file);

    /**
     * @brief Get the number of the sources
     */
    uint32_t nSources() const;

    /**
     * @brief List sensor ids in a string, separated by ';'
     */
    std::string listSensorIds() const;

    /**
     * @brief List sensor names in a string, separated by ';'
     */
    std::string listSensorNames() const;

    /**
     * @brief List sensor uris in a string, separated by ';'
     */
    std::string listUris() const;

    /**
     * @brief Get sensor information for a specific source
     */
    SensorInfo getSensorInfo(uint32_t index) const;

    /**
     * @brief Get properties for setting the source bin(s)
     */
    const YAML::Node &getProperties() const;

    /**
     * @brief If using the nvmultiurisrcbin
     */
    bool useMultiUriSrcBin() const;

protected:
    std::vector<SensorInfo> sensor_info_;
    YAML::Node properties_;
    bool use_nvmultiurisrcbin_;
};
} // namespace deepstream
#endif