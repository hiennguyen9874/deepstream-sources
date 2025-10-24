#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include "check.hpp"

namespace nv {

class EventTimer {
public:
    EventTimer()
    {
        checkRuntime(cudaEventCreate(&begin_));
        checkRuntime(cudaEventCreate(&end_));
    }

    virtual ~EventTimer()
    {
        checkRuntime(cudaEventDestroy(begin_));
        checkRuntime(cudaEventDestroy(end_));
    }

    void start(cudaStream_t stream) { checkRuntime(cudaEventRecord(begin_, stream)); }

    float stop(const char *prefix = "timer")
    {
        float times = 0;
        checkRuntime(cudaEventRecord(end_, stream_));
        checkRuntime(cudaEventSynchronize(end_));
        checkRuntime(cudaEventElapsedTime(&times, begin_, end_));
        printf("[⏰ %s]: \t%.5f ms\n", prefix, times);
        return times;
    }

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t begin_ = nullptr, end_ = nullptr;
};

}; // namespace nv

#endif // __TIMER_HPP__