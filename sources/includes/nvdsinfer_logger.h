#ifndef __NVDSINFER_LOGGER_H__
#define __NVDSINFER_LOGGER_H__

#include <NvInfer.h>
#include <nvdsinfer.h>

#include <memory>
#include <mutex>

#if defined(NDEBUG)
#define INFER_LOG_FORMAT_(fmt) fmt
#else
#define INFER_LOG_FORMAT_(fmt) "%s:%d " fmt, __FILE__, __LINE__
#endif

#define dsInferError(fmt, ...)                                                         \
    do {                                                                               \
        dsInferLogPrint__(NVDSINFER_LOG_ERROR, INFER_LOG_FORMAT_(fmt), ##__VA_ARGS__); \
    } while (0)

#define dsInferWarning(fmt, ...)                                                         \
    do {                                                                                 \
        dsInferLogPrint__(NVDSINFER_LOG_WARNING, INFER_LOG_FORMAT_(fmt), ##__VA_ARGS__); \
    } while (0)

#define dsInferInfo(fmt, ...)                                                         \
    do {                                                                              \
        dsInferLogPrint__(NVDSINFER_LOG_INFO, INFER_LOG_FORMAT_(fmt), ##__VA_ARGS__); \
    } while (0)

#define dsInferDebug(fmt, ...)                                                         \
    do {                                                                               \
        dsInferLogPrint__(NVDSINFER_LOG_DEBUG, INFER_LOG_FORMAT_(fmt), ##__VA_ARGS__); \
    } while (0)

namespace nvdsinfer {
void dsInferLogPrint__(NvDsInferLogLevel level, const char *fmt, ...);
}

extern std::unique_ptr<nvinfer1::ILogger> gTrtLogger;
#endif
