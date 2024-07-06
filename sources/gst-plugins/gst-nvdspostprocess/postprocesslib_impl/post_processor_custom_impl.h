#ifndef __POST_PROCESSOR_CUSTOM_IMPL_HPP__
#define __POST_PROCESSOR_CUSTOM_IMPL_HPP__
#include "post_processor_struct.h"

/*
 * C interfaces
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Type definition for the custom bounding box parsing function.
 *
 * @param[in]  outputLayersInfo A vector containing information on the output
 *                              layers of the model.
 * @param[in]  networkInfo      Network information.
 * @param[in]  detectionParams  Detection parameters required for parsing
 *                              objects.
 * @param[out] objectList       A reference to a vector in which the function
 *                              is to add parsed objects.
 */
typedef bool (*NvDsPostProcessParseCustomFunc)(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsPostProcessParseDetectionParams const &detectionParams,
    std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);

/**
 * Validates a custom parser function definition. Must be called
 * after defining the function.
 */
#define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(customParseFunc)                                         \
    static void checkFunc_##customParseFunc(NvDsPostProcessParseCustomFunc func = customParseFunc) \
    {                                                                                              \
        checkFunc_##customParseFunc();                                                             \
    };                                                                                             \
    extern "C" bool customParseFunc(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,       \
                                    NvDsInferNetworkInfo const &networkInfo,                       \
                                    NvDsPostProcessParseDetectionParams const &detectionParams,    \
                                    std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);

/**
 * Type definition for the custom bounding box and instance mask parsing function.
 *
 * @param[in]  outputLayersInfo A vector containing information on the output
 *                              layers of the model.
 * @param[in]  networkInfo      Network information.
 * @param[in]  detectionParams  Detection parameters required for parsing
 *                              objects.
 * @param[out] objectList       A reference to a vector in which the function
 *                              is to add parsed objects and instance mask.
 */
typedef bool (*NvDsPostProcessInstanceMaskParseCustomFunc)(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsPostProcessParseDetectionParams const &detectionParams,
    std::vector<NvDsPostProcessInstanceMaskInfo> &objectList);

/**
 * Validates a custom parser function definition. Must be called
 * after defining the function.
 */
#define CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(customParseFunc)                        \
    static void checkFunc_##customParseFunc(NvDsPostProcessInstanceMaskParseCustomFunc func =   \
                                                customParseFunc)                                \
    {                                                                                           \
        checkFunc_##customParseFunc();                                                          \
    };                                                                                          \
    extern "C" bool customParseFunc(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,    \
                                    NvDsInferNetworkInfo const &networkInfo,                    \
                                    NvDsPostProcessParseDetectionParams const &detectionParams, \
                                    std::vector<NvDsPostProcessInstanceMaskInfo> &objectList);

/**
 * Type definition for the custom classifier output parsing function.
 *
 * @param[in]  outputLayersInfo  A vector containing information on the
 *                               output layers of the model.
 * @param[in]  networkInfo       Network information.
 * @param[in]  classifierThreshold
                                 Classification confidence threshold.
 * @param[out] attrList          A reference to a vector in which the function
 *                               is to add the parsed attributes.
 * @param[out] descString        A reference to a string object in which the
 *                               function may place a description string.
 */
typedef bool (*NvDsPostProcessClassiferParseCustomFunc)(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsPostProcessAttribute> &attrList,
    std::string &descString);

/**
 * Validates the classifier custom parser function definition. Must be called
 * after defining the function.
 */
#define CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(customParseFunc)                      \
    static void checkFunc_##customParseFunc(NvDsPostProcessClassiferParseCustomFunc func = \
                                                customParseFunc)                           \
    {                                                                                      \
        checkFunc_##customParseFunc();                                                     \
    };                                                                                     \
    extern "C" bool customParseFunc(                                                       \
        std::vector<NvDsInferLayerInfo> const &outputLayersInfo,                           \
        NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,                \
        std::vector<NvDsPostProcessAttribute> &attrList, std::string &descString);

#ifdef __cplusplus
}
#endif
#endif
