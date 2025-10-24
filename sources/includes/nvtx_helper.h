/**
 * @file nvtx_helper.h
 * @brief Helper library for setting NVTX markers
 *
 */
#ifndef __NVTX_HELPER_H__
#define __NVTX_HELPER_H__
#ifdef __cplusplus
extern "C" {
#endif
/**
 * Function definition for pushing/popping a NVTX range
 *
 * @param[in] context If specified, calls nvtxRangePushA().
 *  If not specified(NULL), nvtxRangePop() gets called.
 *
 */
void nvtx_helper_push_pop(char *context);
/**
 * Function definition for starting/stopping a NVTX range
 *
 * @param[in] context If specified, calls nvtxRangeStartA().
 *  If not specified (NULL), nvtxRangeEnd() gets called.
 * @param[in] id The unique ID used to correlate a pair of Start and End events.
 *
 */
void nvtx_helper_start_end(char *context, unsigned long *id);
#ifdef __cplusplus
}
#endif
#endif
