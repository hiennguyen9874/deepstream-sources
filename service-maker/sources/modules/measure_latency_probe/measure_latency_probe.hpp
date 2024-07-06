#include <chrono>
#include <iostream>
#include <vector>

#include "buffer_probe.hpp"

using namespace std;

namespace deepstream {

class NvDsMeasureLatency : public BufferProbe::IBufferObserver {
public:
    virtual probeReturn handleBuffer(BufferProbe &probe, const Buffer &buffer)
    {
        auto latency_info = buffer.measureLatency();
        for (auto &latency : latency_info) {
            cout << "Source id = " << latency.source_id << " Frame_num = " << latency.frame_num
                 << " Frame latency = " << latency.latency << " (ms)" << endl;
        }
        return probeReturn::Probe_Ok;
    }
};

} // namespace deepstream