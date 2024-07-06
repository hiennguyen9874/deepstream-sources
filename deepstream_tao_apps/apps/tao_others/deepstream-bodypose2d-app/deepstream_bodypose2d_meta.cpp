#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "ds_bodypose2d_meta.h"
#include "gstnvdsmeta.h"
#include "nvds_analytics_meta.h"

static gpointer nvds_copy_2dpose_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDs2DposeMetaData *p_2dpose_meta_data = (NvDs2DposeMetaData *)user_meta->user_meta_data;
    NvDs2DposeMetaData *pnew_2dpose_meta_data =
        (NvDs2DposeMetaData *)g_memdup(p_2dpose_meta_data, sizeof(NvDs2DposeMetaData));
    return (gpointer)pnew_2dpose_meta_data;
}

static void nvds_release_2dpose_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDs2DposeMetaData *p_2dpose_meta_data = (NvDs2DposeMetaData *)user_meta->user_meta_data;
    delete p_2dpose_meta_data;
}

/* Bodypose2D model outputs 17 bodyparts coordinates */
extern "C" gboolean nvds_add_2dpose_meta(NvDsBatchMeta *batch_meta,
                                         NvDsObjectMeta *obj_meta,
                                         cvcore::bodypose2d::Human human,
                                         int frame_width,
                                         int frame_height,
                                         float left_offset,
                                         float top_offset)
{
    int parts_num = human.body_parts.size();

    NvDsUserMeta *user_meta = NULL;
    user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    NvDsMetaType user_meta_type = (NvDsMetaType)NVDS_USER_RIVA_META_2DPOSE;
    NvDs2DposeMetaData *p_2dpose_meta_data = new NvDs2DposeMetaData;

    for (int n = 0; n < parts_num; n++) {
        if (n >= BODYPART_TOTAL_NUM)
            break;
        cvcore::bodypose2d::BodyPart body_part = human.body_parts[n];
        if (body_part.loc.x < left_offset) {
            p_2dpose_meta_data->bodypart_locs[n].x = 0;
        } else if (body_part.loc.x > (frame_width - 1 + left_offset)) {
            p_2dpose_meta_data->bodypart_locs[n].x = frame_width - 1;
        } else {
            p_2dpose_meta_data->bodypart_locs[n].x = body_part.loc.x - left_offset;
        }

        if (body_part.loc.y < top_offset) {
            p_2dpose_meta_data->bodypart_locs[n].y = 0;
        } else if (body_part.loc.y > (frame_height - 1 + top_offset)) {
            p_2dpose_meta_data->bodypart_locs[n].y = frame_height - 1;
        } else {
            p_2dpose_meta_data->bodypart_locs[n].y = body_part.loc.y - top_offset;
        }

        p_2dpose_meta_data->bodypart_locs[n].part_idx = body_part.part_idx;
        p_2dpose_meta_data->bodypart_locs[n].score = body_part.score;
    }

    p_2dpose_meta_data->bodypart_num = parts_num;
    p_2dpose_meta_data->score = human.score;

    user_meta->user_meta_data = (void *)(p_2dpose_meta_data);
    user_meta->base_meta.meta_type = user_meta_type;
    user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)nvds_copy_2dpose_meta;
    user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)nvds_release_2dpose_meta;

    /* We want to add NvDsUserMeta to obj level */
    nvds_add_user_meta_to_obj(obj_meta, user_meta);
    return true;
}
