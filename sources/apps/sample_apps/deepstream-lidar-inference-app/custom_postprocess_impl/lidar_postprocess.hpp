#ifndef LIDAR_POST_PROCESS_H_
#define LIDAR_POST_PROCESS_H_
#include <vector>

#include "ds3d/common/ds3d_analysis_datatype.h"

using namespace ds3d;

int ParseCustomBatchedNMS(std::vector<Lidar3DBbox> bndboxes,
                          const float nms_thresh,
                          std::vector<Lidar3DBbox> &nms_pred,
                          const int pre_nms_top_n);

#endif