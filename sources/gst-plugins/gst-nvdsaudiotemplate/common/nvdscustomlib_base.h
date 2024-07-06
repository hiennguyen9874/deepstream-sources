#ifndef __NVDSCUSTOMLIB_BASE_HPP__
#define __NVDSCUSTOMLIB_BASE_HPP__

#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>

#include "nvdscustomlib_interface.hpp"

/* Buffer Pool Configuration Parameters */
struct BufferPoolConfig {
    gint cuda_mem_type;
    guint gpu_id;
    guint max_buffers;
    gint batch_size;
};

class DSCustomLibraryBase : public IDSCustomLibrary {
public:
    DSCustomLibraryBase();

    /* Set Init Parameters */
    virtual bool SetInitParams(DSCustom_CreateParams *params);

    virtual ~DSCustomLibraryBase();

    virtual bool HandleEvent(GstEvent *event) = 0;

    /* Set Custom Properties  of the library */
    virtual bool SetProperty(Property &prop) = 0;
    // TODO: Add getProperty as well

    /* Get GetCompatibleOutputCaps */
    virtual GstCaps *GetCompatibleCaps(GstPadDirection direction,
                                       GstCaps *in_caps,
                                       GstCaps *othercaps);

    /* Process Incoming Buffer */
    virtual BufferResult ProcessBuffer(GstBuffer *inbuf) = 0;

public:
    /* Gstreamer dsexaple2 plugin's base class reference */
    GstBaseTransform *m_element;

    /** GPU ID on which we expect to execute the algorithm */
    guint m_gpuId;

    /* Audio Information */
    GstAudioInfo m_inAudioInfo;
    GstAudioInfo m_outAudioInfo;

    /* Audio Format Information */
    GstAudioFormat m_inAudioFmt;
    GstAudioFormat m_outAudioFmt;

    /* Gst Caps Information */
    GstCaps *m_inCaps;
    GstCaps *m_outCaps;
};

#endif
