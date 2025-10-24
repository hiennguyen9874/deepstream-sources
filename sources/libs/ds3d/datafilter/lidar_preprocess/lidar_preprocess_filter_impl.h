#ifndef DS3D_DATAFILTER_LIDAR_PREPROCESS_FILTER_IMPL_H
#define DS3D_DATAFILTER_LIDAR_PREPROCESS_FILTER_IMPL_H

#include "ds3d/common/helper/cuda_utils.h"
#include "ds3d/common/helper/safe_queue.h"
#include "ds3d/common/hpp/datafilter.hpp"
#include "ds3d/common/impl/impl_datafilter.h"
#include "lidar_preprocess_config.h"
#include "lidar_preprocess_filter.h"
#include "voxelization.hpp"

namespace ds3d {
namespace impl {
namespace filter {

class LidarPreprocessFilter : public BaseImplDataFilter {
public:
    LidarPreprocessFilter() = default;
    ~LidarPreprocessFilter() override;

protected:
    ErrCode processImpl(GuardDataMap datamap,
                        OnGuardDataCBImpl outputDataCb,
                        OnGuardDataCBImpl inputConsumedCb) override;
    ErrCode stopImpl() override;
    ErrCode flushImpl() override;
    ErrCode startImpl(const std::string &content, const std::string &path);

private:
    ErrCode reserveInputMem(uint &devId, uint32_t count, int &batchSize);
    ErrCode doLidarPreProcess(GuardDataMap &dataMap, SharedBatchArray &batchArray);

    Config _config;
    std::unordered_map<std::string, std::shared_ptr<BufferPool<UniqCudaTensorBuf>>>
        _inputBuferPoolMap;
    volatile bool _inProcess = false;
    Ptr<CudaStream> _cudaStream;
    std::unique_ptr<bevfusion::pointpillars::Voxelization> _voxelization;
};

} // namespace filter
} // namespace impl
} // namespace ds3d

#endif // DS3D_DATAFILTER_LIDAR_PREPROCESS_FILTER_IMPL_H
