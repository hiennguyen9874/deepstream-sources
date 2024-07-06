#ifndef __NVSTREAMMUX_DEBUG_H__
#define __NVSTREAMMUX_DEBUG_H__

class INvStreammuxDebug {
public:
    virtual void DebugPrint(const char *format, ...) = 0;
};

#endif /**< __NVSTREAMMUX_DEBUG_H__ */
