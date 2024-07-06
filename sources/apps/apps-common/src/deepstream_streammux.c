#include "deepstream_streammux.h"

#include <string.h>

#include "deepstream_common.h"

// Create bin, add queue and the element, link all elements and ghost pads,
// Set the element properties from the parsed config
gboolean set_streammux_properties(NvDsStreammuxConfig *config, GstElement *element)
{
    gboolean ret = FALSE;
    const gchar *new_mux_str = g_getenv("USE_NEW_NVSTREAMMUX");
    gboolean use_new_mux = !g_strcmp0(new_mux_str, "yes");

    if (!use_new_mux) {
        g_object_set(G_OBJECT(element), "gpu-id", config->gpu_id, NULL);

        g_object_set(G_OBJECT(element), "nvbuf-memory-type", config->nvbuf_memory_type, NULL);

        g_object_set(G_OBJECT(element), "live-source", config->live_source, NULL);

        g_object_set(G_OBJECT(element), "batched-push-timeout", config->batched_push_timeout, NULL);

        if (config->buffer_pool_size >= 4) {
            g_object_set(G_OBJECT(element), "buffer-pool-size", config->buffer_pool_size, NULL);
        }

        g_object_set(G_OBJECT(element), "enable-padding", config->enable_padding, NULL);

        if (config->pipeline_width && config->pipeline_height) {
            g_object_set(G_OBJECT(element), "width", config->pipeline_width, NULL);
            g_object_set(G_OBJECT(element), "height", config->pipeline_height, NULL);
        }
    }

    if (config->batch_size) {
        g_object_set(G_OBJECT(element), "batch-size", config->batch_size, NULL);
    }

    g_object_set(G_OBJECT(element), "attach-sys-ts", config->attach_sys_ts_as_ntp, NULL);

    if (config->config_file_path) {
        g_object_set(G_OBJECT(element), "config-file-path", GET_FILE_PATH(config->config_file_path),
                     NULL);
    }

    g_object_set(G_OBJECT(element), "frame-num-reset-on-stream-reset",
                 config->frame_num_reset_on_stream_reset, NULL);

    g_object_set(G_OBJECT(element), "sync-inputs", config->sync_inputs, NULL);

    g_object_set(G_OBJECT(element), "max-latency", config->max_latency, NULL);
    g_object_set(G_OBJECT(element), "frame-num-reset-on-eos", config->frame_num_reset_on_eos, NULL);

    ret = TRUE;

    return ret;
}
