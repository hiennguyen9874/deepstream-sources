#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "ds_facialmark_meta.h"
#include "gstnvdsmeta.h"
#include "nvds_analytics_meta.h"

/*Faciallandmark metadata functions*/

static gpointer nvds_copy_facemark_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsFacePointsMetaData *p_facemark_meta_data =
        (NvDsFacePointsMetaData *)user_meta->user_meta_data;
    NvDsFacePointsMetaData *pnew_facemark_meta_data =
        (NvDsFacePointsMetaData *)g_memdup(p_facemark_meta_data, sizeof(NvDsFacePointsMetaData));
    return (gpointer)pnew_facemark_meta_data;
}

static void nvds_release_facemark_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsFacePointsMetaData *p_facemark_meta_data =
        (NvDsFacePointsMetaData *)user_meta->user_meta_data;
    delete p_facemark_meta_data;
}

/* Faciallandmark model outputs 80 facial landmark coordinates, the right eye */
/* are marked by the coordinates from 36 to 41, and left eye are marked by the*/
/* coordinates from 42 to 47                                                  */
#define LEFT_EYE_MARKS_START_INDEX 42
#define LEFT_EYE_MARKS_END_INDEX 47
#define RIGHT_EYE_MARKS_START_INDEX 36
#define RIGHT_EYE_MARKS_END_INDEX 41
extern "C" gboolean nvds_add_facemark_meta(
    NvDsBatchMeta *batch_meta,
    NvDsObjectMeta *obj_meta,
    cvcore::ArrayN<cvcore::Vector2f,
                   cvcore::faciallandmarks::FacialLandmarks::MAX_NUM_FACIAL_LANDMARKS> &marks,
    float *confidence)
{
    NvDsUserMeta *user_meta = NULL;
    user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    NvDsMetaType user_meta_type = (NvDsMetaType)NVDS_USER_RIVA_META_FACEMARK;
    NvDsFacePointsMetaData *p_facemark_meta_data = new NvDsFacePointsMetaData;

    int marks_num = marks.getSize();

    for (int n = 0; n < marks_num; n++) {
        if (marks[n].x < 0.0) {
            p_facemark_meta_data->mark[n].x = 0.0;
        } else if (marks[n].x >= obj_meta->rect_params.width) {
            p_facemark_meta_data->mark[n].x = obj_meta->rect_params.width - 1;
        } else {
            p_facemark_meta_data->mark[n].x = marks[n].x;
        }

        if (marks[n].y < 0.0) {
            p_facemark_meta_data->mark[n].y = 0.0;
        } else if (marks[n].y >= obj_meta->rect_params.height) {
            p_facemark_meta_data->mark[n].y = obj_meta->rect_params.height - 1;
        } else {
            p_facemark_meta_data->mark[n].y = marks[n].y;
        }
        p_facemark_meta_data->mark[n].score = confidence[n];
        if (n == RIGHT_EYE_MARKS_START_INDEX) {
            p_facemark_meta_data->right_eye_rect.left = p_facemark_meta_data->mark[n].x;
            p_facemark_meta_data->right_eye_rect.top = p_facemark_meta_data->mark[n].y;
            p_facemark_meta_data->right_eye_rect.right = p_facemark_meta_data->mark[n].x;
            p_facemark_meta_data->right_eye_rect.bottom = p_facemark_meta_data->mark[n].y;
        } else if (n == LEFT_EYE_MARKS_START_INDEX) {
            p_facemark_meta_data->left_eye_rect.left = p_facemark_meta_data->mark[n].x;
            p_facemark_meta_data->left_eye_rect.top = p_facemark_meta_data->mark[n].y;
            p_facemark_meta_data->left_eye_rect.right = p_facemark_meta_data->mark[n].x;
            p_facemark_meta_data->left_eye_rect.bottom = p_facemark_meta_data->mark[n].y;
        } else if (n > RIGHT_EYE_MARKS_START_INDEX && n <= RIGHT_EYE_MARKS_END_INDEX) {
            if (p_facemark_meta_data->right_eye_rect.left > p_facemark_meta_data->mark[n].x)
                p_facemark_meta_data->right_eye_rect.left = p_facemark_meta_data->mark[n].x;
            if (p_facemark_meta_data->right_eye_rect.top > p_facemark_meta_data->mark[n].y)
                p_facemark_meta_data->right_eye_rect.top = p_facemark_meta_data->mark[n].y;
            if (p_facemark_meta_data->right_eye_rect.right < p_facemark_meta_data->mark[n].x)
                p_facemark_meta_data->right_eye_rect.right = p_facemark_meta_data->mark[n].x;
            if (p_facemark_meta_data->right_eye_rect.bottom < p_facemark_meta_data->mark[n].y)
                p_facemark_meta_data->right_eye_rect.bottom = p_facemark_meta_data->mark[n].y;
        } else if (n > LEFT_EYE_MARKS_START_INDEX && n <= LEFT_EYE_MARKS_END_INDEX) {
            if (p_facemark_meta_data->left_eye_rect.left > p_facemark_meta_data->mark[n].x)
                p_facemark_meta_data->left_eye_rect.left = p_facemark_meta_data->mark[n].x;
            if (p_facemark_meta_data->left_eye_rect.top > p_facemark_meta_data->mark[n].y)
                p_facemark_meta_data->left_eye_rect.top = p_facemark_meta_data->mark[n].y;
            if (p_facemark_meta_data->left_eye_rect.right < p_facemark_meta_data->mark[n].x)
                p_facemark_meta_data->left_eye_rect.right = p_facemark_meta_data->mark[n].x;
            if (p_facemark_meta_data->left_eye_rect.bottom < p_facemark_meta_data->mark[n].y)
                p_facemark_meta_data->left_eye_rect.bottom = p_facemark_meta_data->mark[n].y;
        }
    }

    p_facemark_meta_data->facemark_num = marks_num;

    user_meta->user_meta_data = (void *)(p_facemark_meta_data);
    user_meta->base_meta.meta_type = user_meta_type;
    user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)nvds_copy_facemark_meta;
    user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)nvds_release_facemark_meta;

    /* We want to add NvDsUserMeta to obj level */
    nvds_add_user_meta_to_obj(obj_meta, user_meta);
    return true;
}
