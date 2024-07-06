#include "deepstream_common.h"
#include "deepstream_tracker.h"

GST_DEBUG_CATEGORY_EXTERN(NVDS_APP);

gboolean create_tracking_bin(NvDsTrackerConfig *config, NvDsTrackerBin *bin)
{
    gboolean ret = FALSE;

    bin->bin = gst_bin_new("tracking_bin");
    if (!bin->bin) {
        NVGSTDS_ERR_MSG_V("Failed to create 'tracking_bin'");
        goto done;
    }

    bin->tracker = gst_element_factory_make(NVDS_ELEM_TRACKER, "tracking_tracker");
    if (!bin->tracker) {
        NVGSTDS_ERR_MSG_V("Failed to create 'tracking_tracker'");
        goto done;
    }

    g_object_set(G_OBJECT(bin->tracker), "tracker-width", config->width, "tracker-height",
                 config->height, "gpu-id", config->gpu_id, "ll-config-file", config->ll_config_file,
                 "ll-lib-file", config->ll_lib_file, NULL);

    if (config->batch_config_set) {
        g_object_set(G_OBJECT(bin->tracker), "enable-batch-process", config->enable_batch_process,
                     NULL);
    }

    g_object_set(G_OBJECT(bin->tracker), "enable-past-frame", config->enable_past_frame, NULL);

    g_object_set(G_OBJECT(bin->tracker), "display-tracking-id", config->display_tracking_id, NULL);

    g_object_set(G_OBJECT(bin->tracker), "tracking-surface-type", config->tracking_surface_type,
                 NULL);

    gst_bin_add_many(GST_BIN(bin->bin), bin->tracker, NULL);

    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->tracker, "sink");

    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->tracker, "src");

    ret = TRUE;

    GST_CAT_DEBUG(NVDS_APP, "Tracker bin created successfully");

done:
    if (!ret) {
        NVGSTDS_ERR_MSG_V("%s failed", __func__);
    }
    return ret;
}
