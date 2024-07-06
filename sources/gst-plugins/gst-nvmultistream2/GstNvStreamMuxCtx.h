/**
 * @file GstNvStreamMuxCtx.h
 * @brief  StreamMux heler context class
 */

#ifndef _GST_NVSTREAMMUXCTX_H_
#define _GST_NVSTREAMMUXCTX_H_

#include <mutex>
#include <unordered_map>

#include "nvbufaudio.h"

class GstNvStreamMuxCtx {
public:
    GstNvStreamMuxCtx();
    void SaveAudioParams(uint32_t padId, uint32_t sourceId, NvBufAudioParams audioParams);
    NvBufAudioParams GetAudioParams(uint32_t padId);
    void SetMemTypeNVMM(uint32_t padId, bool isNVMM);
    bool IsMemTypeNVMM(uint32_t padId);

private:
    std::mutex mutex;
    std::unordered_map<uint32_t, NvBufAudioParams> audioParamsMap;
    std::unordered_map<uint32_t, bool> isNVMMMap;
};

#endif /**< _GST_NVSTREAMMUXCTX_H_ */
