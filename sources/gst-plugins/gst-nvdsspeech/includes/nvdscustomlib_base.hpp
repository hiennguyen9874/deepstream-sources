#ifndef __NVDS_SPEECH_CUSTOMLIB_BASE_HPP__
#define __NVDS_SPEECH_CUSTOMLIB_BASE_HPP__

#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>

#include "nvdscustomlib_interface.hpp"

namespace nvdsspeech {

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
    /* Gstreamer dsspeech plugin's base class reference */
    GstBaseTransform *m_element{nullptr};
    /* Gst Caps Information */
    GstCaps *m_inCaps{nullptr};
    GstCaps *m_outCaps{nullptr};
    std::string m_configFile;

    /* Audio Information */
    GstAudioInfo m_inAudioInfo{nullptr, GST_AUDIO_FLAG_NONE};
    GstAudioFormat m_inAudioFmt = GST_AUDIO_FORMAT_UNKNOWN;
    /* Output Information */
    CapsType m_OutType = CapsType::kNone;
    GstAudioInfo m_outAudioInfo{nullptr, GST_AUDIO_FLAG_NONE};
};

bool DSCustomLibraryBase::SetProperty(const Property &prop)
{
    if (prop.key == NVDS_CONFIG_FILE_PROPERTY) {
        m_configFile = prop.value;
    }
    return true;
}

bool DSCustomLibraryBase::Initialize()
{
    return true;
}

bool DSCustomLibraryBase::StartWithParams(DSCustom_CreateParams *params)
{
    assert(params);
    m_element = params->m_element;
    m_inCaps = gst_caps_copy(params->m_inCaps);
    m_outCaps = gst_caps_copy(params->m_outCaps);

    gst_audio_info_from_caps(&m_inAudioInfo, m_inCaps);

    GstStructure *outStr = gst_caps_get_structure(m_outCaps, 0);
    if (gst_structure_has_name(outStr, "audio/x-raw")) {
        m_OutType = CapsType::kAudio;
        gst_audio_info_from_caps(&m_outAudioInfo, m_outCaps);
    } else {
        m_OutType = CapsType::kText;
    }

    m_inAudioFmt = GST_AUDIO_FORMAT_INFO_FORMAT(m_inAudioInfo.finfo);

    return true;
}

DSCustomLibraryBase::~DSCustomLibraryBase()
{
    if (m_inCaps) {
        gst_caps_unref(m_inCaps);
    }
    if (m_outCaps) {
        gst_caps_unref(m_outCaps);
    }
}

GstCaps *DSCustomLibraryBase::GetCompatibleCaps(GstPadDirection direction,
                                                GstCaps *inCaps,
                                                GstCaps *otherCaps)
{
    if (!otherCaps) {
        g_print(
            "WARNING: DSCustomLibraryBase detect empty otherCaps, will try "
            "inCaps");
        return gst_caps_ref(inCaps);
    }

    GstCaps *result = nullptr;
    if (!gst_caps_is_fixed(otherCaps)) {
        result = gst_caps_fixate(otherCaps);
        return result;
    } else {
        result = gst_caps_ref(otherCaps);
    }
    return result;
}

} // namespace nvdsspeech

#endif
