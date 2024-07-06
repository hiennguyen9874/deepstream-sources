#ifndef __NVDSCUSTOMLIB_INTERFACE_HPP__
#define __NVDSCUSTOMLIB_INTERFACE_HPP__

#include <cuda_runtime.h>
#include <gst/gstbuffer.h>
#include <gst/video/video.h>

#include <string>

#include "../gstaudio2video.h"

enum class BufferResult {
    Buffer_Ok,    // Push the buffer from submit_input function
    Buffer_Drop,  // Drop the buffer inside submit_input function
    Buffer_Async, // Return from submit_input function, custom lib to push the buffer
    Buffer_Error  // Error occured
};

struct DSCustom_CreateParams {
    GstElement *m_element;
};

struct Property {
    Property(std::string arg_key, std::string arg_value) : key(arg_key), value(arg_value) {}

    std::string key;
    std::string value;
};

class IDSCustomLibrary {
public:
    virtual bool SetInitParams(DSCustom_CreateParams *params) = 0;
    virtual bool SetProperty(Property &prop) = 0;
    virtual bool HandleEvent(GstEvent *event) = 0;
    virtual BufferResult ProcessBuffer(GstAudio2Video *base,
                                       GstBuffer *audio,
                                       GstVideoFrame *video) = 0;
    virtual char *QueryProperties() = 0;
    virtual ~IDSCustomLibrary(){};
};

#endif
