#ifndef __NVDSPOSTPROCESSLIB_INTERFACE_HPP__
#define __NVDSPOSTPROCESSLIB_INTERFACE_HPP__

#include <gst/gstbuffer.h>

#include <string>

enum class BufferResult {
    Buffer_Ok,    // Push the buffer from submit_input function
    Buffer_Drop,  // Drop the buffer inside submit_input function
    Buffer_Async, // Return from submit_input function, postprocess lib to push the buffer
    Buffer_Error  // Error occured
};

struct DSPostProcess_CreateParams {
    GstBaseTransform *m_element;
    guint m_gpuId;
    cudaStream_t m_cudaStream;
    bool m_preprocessor_support;
};

struct Property {
    Property(std::string arg_key, std::string arg_value) : key(arg_key), value(arg_value) {}

    std::string key;
    std::string value;
};

class IDSPostProcessLibrary {
public:
    virtual bool HandleEvent(GstEvent *event) = 0;
    virtual bool SetConfigFile(const gchar *config_file) = 0;
    virtual BufferResult ProcessBuffer(GstBuffer *inbuf) = 0;
    virtual ~IDSPostProcessLibrary() {};
};

#endif
