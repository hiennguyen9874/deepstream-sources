#ifndef CVCORE_MODEL_H
#define CVCORE_MODEL_H

#include <vector>

#include "Image.h"

namespace cvcore {

/**
 * Struct to describe input type required by the model
 */
struct ModelInputParams {
    size_t maxBatchSize;      /**< maxbatchSize supported by network*/
    size_t inputLayerWidth;   /**< Input layer width */
    size_t inputLayerHeight;  /**< Input layer Height */
    ImageType modelInputType; /**< Input Layout type */
};

/**
 * Struct to describe the model
 */
struct ModelInferenceParams {
    std::string engineFilePath;            /**< Engine file path. */
    std::vector<std::string> inputLayers;  /**< names of input layers. */
    std::vector<std::string> outputLayers; /**< names of output layers. */
};

} // namespace cvcore

#endif // CVCORE_MODEL_H
