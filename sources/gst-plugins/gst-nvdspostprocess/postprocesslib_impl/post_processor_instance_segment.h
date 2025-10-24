#ifndef __POST_PROCESSOR_INSTANCE_SEGMENT_HPP__
#define __POST_PROCESSOR_INSTANCE_SEGMENT_HPP__

#include "post_processor.h"

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

class InstanceSegmentModelPostProcessor : public ModelPostProcessor {
public:
    InstanceSegmentModelPostProcessor(int id, int gpuId = 0)
        : ModelPostProcessor(NvDsPostProcessNetworkType_InstanceSegmentation, id, gpuId)
    {
    }

    NvDsPostProcessStatus initResource(NvDsPostProcessContextInitParams &initParams) override;
    ~InstanceSegmentModelPostProcessor() override = default;
    NvDsPostProcessStatus parseEachFrame(const std::vector<NvDsInferLayerInfo> &outputLayers,
                                         NvDsPostProcessFrameOutput &result) override;
    void attachMetadata(NvBufSurface *surf,
                        gint batch_idx,
                        NvDsBatchMeta *batch_meta,
                        NvDsFrameMeta *frame_meta,
                        NvDsObjectMeta *object_meta,
                        NvDsObjectMeta *parent_obj_meta,
                        NvDsPostProcessFrameOutput &detection_output,
                        NvDsPostProcessDetectionParams *all_params,
                        std::set<gint> &filterOutClassIds,
                        int32_t unique_id,
                        gboolean output_instance_mask,
                        gboolean process_full_frame,
                        float segmentationThreshold,
                        gboolean maintain_aspect_ratio,
                        NvDsRoiMeta *roi_meta,
                        gboolean symmetric_padding) override;

    void releaseFrameOutput(NvDsPostProcessFrameOutput &frameOutput) override;

private:
    void fillUnclusteredOutput(NvDsPostProcessDetectionOutput &output);
    NvDsPostProcessStatus fillDetectionOutput(const std::vector<NvDsInferLayerInfo> &outputLayers,
                                              NvDsPostProcessDetectionOutput &output);
    void preClusteringThreshold(NvDsPostProcessParseDetectionParams const &detectionParams,
                                std::vector<NvDsPostProcessInstanceMaskInfo> &objectList);
    void filterTopKOutputs(int const topK,
                           std::vector<NvDsPostProcessInstanceMaskInfo> &objectList);

private:
    NvDsPostProcessClusterMode m_ClusterMode;

    uint32_t m_NumDetectedClasses = 0;

    std::vector<NvDsPostProcessDetectionParams> m_PerClassDetectionParams;
    NvDsPostProcessParseDetectionParams m_DetectionParams = {0, {}, {}};

    std::vector<NvDsPostProcessInstanceMaskInfo> m_InstanceMaskList;
    /* Vector of NvDsPostProcessInstanceMaskInfo vectors for each class. */
    std::vector<std::vector<NvDsPostProcessInstanceMaskInfo>> m_PerClassInstanceMaskList;

    NvDsPostProcessInstanceMaskParseCustomFunc m_CustomParseFunc = nullptr;
};

#endif
