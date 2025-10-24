#ifndef __NVDSCUSTOMLIB_INTERFACE_HPP__
#define __NVDSCUSTOMLIB_INTERFACE_HPP__

#include <gst/gstbuffer.h>

#include <string>

enum class BufferResult {
    Buffer_Ok,    // Push the buffer from submit_input function
    Buffer_Drop,  // Drop the buffer inside submit_input function
    Buffer_Async, // Return from submit_input function, custom lib to push the buffer
    Buffer_Error  // Error occured
};

struct DSCustom_CreateParams {
    GstBaseTransform *m_element;
    GstCaps *m_inCaps;
    GstCaps *m_outCaps;
    guint m_gpuId;
    guint m_batchSize;
};

struct Property {
    Property(std::string arg_key, std::string arg_value) : key(arg_key), value(arg_value) {}

    std::string key;
    std::string value;
};

class IDSCustomLibrary {
public:
    virtual bool SetInitParams(DSCustom_CreateParams *params) = 0;
    virtual bool HandleEvent(GstEvent *event) = 0;
    virtual char *QueryProperties() = 0;
    virtual bool SetProperty(Property &prop) = 0;
    virtual GstCaps *GetCompatibleCaps(GstPadDirection direction,
                                       GstCaps *in_caps,
                                       GstCaps *othercaps) = 0;
    virtual BufferResult ProcessBuffer(GstBuffer *inbuf) = 0;
    virtual ~IDSCustomLibrary() {};
};

#endif
