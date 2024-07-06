#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "ds_gaze_meta.h"
#include "gstnvdsmeta.h"

/*GazeNet metadata functions*/

static gpointer nvds_copy_gaze_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsGazeMetaData *p_gaze_meta_data = (NvDsGazeMetaData *)user_meta->user_meta_data;
    NvDsGazeMetaData *pnew_gaze_meta_data =
        (NvDsGazeMetaData *)g_memdup(p_gaze_meta_data, sizeof(NvDsGazeMetaData));
    return (gpointer)pnew_gaze_meta_data;
}

static void nvds_release_gaze_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsGazeMetaData *p_gaze_meta_data = (NvDsGazeMetaData *)user_meta->user_meta_data;
    delete p_gaze_meta_data;
}

/* Gazenet model outputs 5 float parameters */
extern "C" gboolean nvds_add_gaze_meta(
    NvDsBatchMeta *batch_meta,
    NvDsObjectMeta *obj_meta,
    cvcore::ArrayN<float, cvcore::gazenet::GazeNet::OUTPUT_SIZE> &params,
    cvcore::Array<cvcore::Vector2i> &leftStart,
    cvcore::Array<cvcore::Vector2i> &leftEnd,
    cvcore::Array<cvcore::Vector2i> &rightStart,
    cvcore::Array<cvcore::Vector2i> &rightEnd)
{
    NvDsUserMeta *user_meta = NULL;
    user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    NvDsMetaType user_meta_type = (NvDsMetaType)NVDS_USER_RIVA_META_GAZE;
    NvDsGazeMetaData *p_gaze_meta_data = new NvDsGazeMetaData;

    int params_num = params.getSize();

    for (int n = 0; n < params_num; n++) {
        p_gaze_meta_data->gaze_params[n] = params[n];
    }
    p_gaze_meta_data->left_start_x = leftStart[0].x;
    p_gaze_meta_data->left_start_y = leftStart[0].y;
    p_gaze_meta_data->left_end_x = leftEnd[0].x;
    p_gaze_meta_data->left_end_y = leftEnd[0].y;
    p_gaze_meta_data->right_start_x = rightStart[0].x;
    p_gaze_meta_data->right_start_y = rightStart[0].y;
    p_gaze_meta_data->right_end_x = rightEnd[0].x;
    p_gaze_meta_data->right_end_y = rightEnd[0].y;

    user_meta->user_meta_data = (void *)(p_gaze_meta_data);
    user_meta->base_meta.meta_type = user_meta_type;
    user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)nvds_copy_gaze_meta;
    user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)nvds_release_gaze_meta;

    /* We want to add NvDsUserMeta to obj level */
    nvds_add_user_meta_to_obj(obj_meta, user_meta);
    return true;
}
