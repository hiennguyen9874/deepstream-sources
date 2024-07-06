#ifndef __DSDIRECTION_LIB__
#define __DSDIRECTION_LIB__
#include "nvds_opticalflow_meta.h"
#include "nvll_osd_struct.h"

#define MAX_LABEL_SIZE 128
#ifdef __cplusplus
extern "C" {
#endif

// Detected/Labelled object structure, stores bounding box info along with label
typedef struct {
    float flowx;
    float flowy;
    char direction[MAX_LABEL_SIZE];
} DsDirectionObject;

// Output data returned after processing
typedef struct {
    DsDirectionObject object;
} DsDirectionOutput;

// Dequeue processed output
DsDirectionOutput *DsDirectionProcess(NvOFFlowVector *in_flow,
                                      int flow_cols,
                                      int flow_rows,
                                      int flow_bsize,
                                      NvOSD_RectParams *rect_param);

#ifdef __cplusplus
}
#endif

#endif
