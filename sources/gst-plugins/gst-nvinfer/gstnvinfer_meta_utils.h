#include "gstnvinfer.h"
#include "gstnvinfer_impl.h"

void attach_metadata_detector(GstNvInfer *nvinfer,
                              GstMiniObject *tensor_out_object,
                              GstNvInferFrame &frame,
                              NvDsInferDetectionOutput &detection_output,
                              float segmentationThreshold);

void attach_metadata_classifier(GstNvInfer *nvinfer,
                                GstMiniObject *tensor_out_object,
                                GstNvInferFrame &frame,
                                GstNvInferObjectInfo &object_info);

void merge_classification_output(GstNvInferObjectHistory &history,
                                 GstNvInferObjectInfo &new_result);

void attach_metadata_segmentation(GstNvInfer *nvinfer,
                                  GstMiniObject *tensor_out_object,
                                  GstNvInferFrame &frame,
                                  NvDsInferSegmentationOutput &segmentation_output);

/* Attaches the raw tensor output to the GstBuffer as metadata. */
void attach_tensor_output_meta(GstNvInfer *nvinfer,
                               GstMiniObject *tensor_out_object,
                               GstNvInferBatch *batch,
                               NvDsInferContextBatchOutput *batch_output);
