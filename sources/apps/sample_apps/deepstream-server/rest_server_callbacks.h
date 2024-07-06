#include "gst-nvcustomevent.h"
#include "gst-nvmultiurisrcbincreator.h"
#include "nvds_appctx_server.h"
#include "nvds_rest_server.h"

/* Callback to handle application related REST API requests*/
void s_appinstance_callback_impl(NvDsAppInstanceInfo *appinstance_info, void *ctx);

/* Callback to handle osd related REST API requests*/
void s_osd_callback_impl(NvDsOsdInfo *osd_info, void *ctx);

/* Callback to handle nvstreammux related REST API requests*/
void s_mux_callback_impl(NvDsMuxInfo *mux_info, void *ctx);

/* Callback to handle encoder specific REST API requests*/
void s_enc_callback_impl(NvDsEncInfo *enc_info, void *ctx);

/* Callback to handle encoder specific REST API requests*/
void s_conv_callback_impl(NvDsConvInfo *conv_info, void *ctx);

/* Callback to handle nvinferserver specific REST API requests*/
void s_inferserver_callback_impl(NvDsInferServerInfo *inferserver_info, void *ctx);

/* Callback to handle nvinfer specific REST API requests*/
void s_infer_callback_impl(NvDsInferInfo *infer_info, void *ctx);

/* Callback to handle nvv4l2decoder specific REST API requests*/
void s_dec_callback_impl(NvDsDecInfo *dec_info, void *ctx);

/* Callback to handle nvdspreprocess specific REST API requests*/
void s_roi_callback_impl(NvDsRoiInfo *roi_info, void *ctx);

/* Callback to handle stream add/remove specific REST API requests*/
void s_stream_callback_impl(NvDsStreamInfo *stream_info, void *ctx);