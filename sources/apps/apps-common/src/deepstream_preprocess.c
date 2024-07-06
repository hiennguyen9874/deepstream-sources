#include "deepstream_preprocess.h"

#include "deepstream_common.h"

// Create bin, add queue and the element, link all elements and ghost pads,
// Set the element properties from the parsed config
gboolean create_preprocess_bin(NvDsPreProcessConfig *config, NvDsPreProcessBin *bin)
{
    gboolean ret = FALSE;

    bin->bin = gst_bin_new("preprocess_bin");
    if (!bin->bin) {
        NVGSTDS_ERR_MSG_V("Failed to create 'preprocess_bin'");
        goto done;
    }

    bin->queue = gst_element_factory_make(NVDS_ELEM_QUEUE, "preprocess_queue");
    if (!bin->queue) {
        NVGSTDS_ERR_MSG_V("Failed to create 'preprocess_queue'");
        goto done;
    }

    bin->preprocess = gst_element_factory_make(NVDS_ELEM_PREPROCESS, "preprocess0");
    if (!bin->preprocess) {
        NVGSTDS_ERR_MSG_V("Failed to create 'preprocess0'");
        goto done;
    }

    gst_bin_add_many(GST_BIN(bin->bin), bin->queue, bin->preprocess, NULL);

    NVGSTDS_LINK_ELEMENT(bin->queue, bin->preprocess);

    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->queue, "sink");

    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->preprocess, "src");

    g_object_set(G_OBJECT(bin->preprocess), "config-file", config->config_file_path, NULL);

    ret = TRUE;

done:
    if (!ret) {
        NVGSTDS_ERR_MSG_V("%s failed", __func__);
    }

    return ret;
}