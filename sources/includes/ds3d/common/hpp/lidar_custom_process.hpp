#ifndef DS3D_COMMON_HPP_LIDAR_CUSTOM_PROCESS_HPP
#define DS3D_COMMON_HPP_LIDAR_CUSTOM_PROCESS_HPP

#include "infer_datatypes.h"
namespace ds3d {

using namespace nvdsinferserver;

class IInferCustomPreprocessor {
public:
    virtual ~IInferCustomPreprocessor() = default;
    virtual NvDsInferStatus preproc(GuardDataMap &dataMap, SharedIBatchArray batchArray) = 0;
};

} // namespace ds3d

#endif // DS3D_COMMON_HPP_LIDAR_CUSTOM_PROCESS_HPP
