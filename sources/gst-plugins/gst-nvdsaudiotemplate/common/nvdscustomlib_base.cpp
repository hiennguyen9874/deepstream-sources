#include "nvdscustomlib_base.h"

DSCustomLibraryBase::DSCustomLibraryBase(GstBaseTransform *btrans)
    : m_element(btrans), m_gpuId(0), m_inCaps(nullptr), m_outCaps(nullptr)
{
}

bool DSCustomLibraryBase::SetInitParams(DSCustom_CreateParams *params)
{
    m_element = params->m_element;
    m_inCaps = params->m_inCaps;
    m_outCaps = params->m_outCaps;
    m_gpuId = params->m_gpuId;

    gst_audio_info_from_caps(&m_inAudioInfo, m_inCaps);
    gst_audio_info_from_caps(&m_outAudioInfo, m_outCaps);

    m_inAudioFmt = GST_AUDIO_FORMAT_INFO_FORMAT(m_inAudioInfo.finfo);
    m_outAudioFmt = GST_AUDIO_FORMAT_INFO_FORMAT(m_outAudioInfo.finfo);

    return true;
}

DSCustomLibraryBase::~DSCustomLibraryBase()
{
}

GstCaps *DSCustomLibraryBase::GetCompatibleCaps(GstPadDirection direction,
                                                GstCaps *in_caps,
                                                GstCaps *othercaps)
{
    GstCaps *result = gst_caps_copy(in_caps);
    return result;
}