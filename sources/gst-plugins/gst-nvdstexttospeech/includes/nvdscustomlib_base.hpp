#ifndef __NVDS_TTS_CUSTOMLIB_BASE_HPP__
#define __NVDS_TTS_CUSTOMLIB_BASE_HPP__

#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>

#include "nvdscustomlib_interface.hpp"

namespace nvdstts {

class DSCustomLibraryBase : public IDSCustomLibrary {
public:
    DSCustomLibraryBase() = default;
    virtual ~DSCustomLibraryBase() override;

    bool Initialize() override;

    /* Set Init Parameters */
    bool StartWithParams(DSCustom_CreateParams *params) override;

    /* Set Each Property */
    bool SetProperty(const Property &prop) override;

    /* Get Compatible Input/Output Caps */
    GstCaps *GetCompatibleCaps(GstPadDirection direction,
                               GstCaps *inCaps,
                               GstCaps *otherCaps) override;

    /* Handle event, e.g. EOS... */
    bool HandleEvent(GstEvent *event) override { return true; }

    /* Process Input Buffer */
    BufferResult ProcessBuffer(GstBuffer *inbuf) override = 0;

protected:
    /* Gstreamer dstts plugin's base class reference */
    GstBaseTransform *m_element{nullptr};
    /* Gst Caps Information */
    GstCaps *m_inCaps{nullptr};
    GstCaps *m_outCaps{nullptr};
    std::string m_configFile;

    /* Audio Information */
    /* Output Information */
    CapsType m_OutType = CapsType::kNone;
    GstAudioInfo m_outAudioInfo{nullptr, GST_AUDIO_FLAG_NONE};
    GstAudioFormat m_outAudioFmt = GST_AUDIO_FORMAT_UNKNOWN;
};

} // namespace nvdstts

#endif
