#include "GstNvStreamMuxCtx.h"

GstNvStreamMuxCtx::GstNvStreamMuxCtx() : mutex(), audioParamsMap()
{
}

void GstNvStreamMuxCtx::SaveAudioParams(uint32_t padId,
                                        uint32_t sourceId,
                                        NvBufAudioParams audioParams)
{
    std::unique_lock<std::mutex> lck(mutex);
    audioParamsMap[padId] = audioParams;
    audioParamsMap[padId].sourceId = sourceId;
    audioParamsMap[padId].padId = padId;
}

NvBufAudioParams GstNvStreamMuxCtx::GetAudioParams(uint32_t padId)
{
    std::unique_lock<std::mutex> lck(mutex);
    return audioParamsMap[padId];
}

void GstNvStreamMuxCtx::SetMemTypeNVMM(uint32_t padId, bool isNVMM)
{
    std::unique_lock<std::mutex> lck(mutex);
    isNVMMMap[padId] = isNVMM;
}

bool GstNvStreamMuxCtx::IsMemTypeNVMM(uint32_t padId)
{
    std::unique_lock<std::mutex> lck(mutex);
    return isNVMMMap[padId];
}
