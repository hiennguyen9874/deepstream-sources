#ifndef DS3D_COMMON_HELPER_SIGNALSHOT_H
#define DS3D_COMMON_HELPER_SIGNALSHOT_H

#include <ds3d/common/common.h>

namespace ds3d {

class SignalShot {
    std::mutex _mutex;
    std::condition_variable _cond;

public:
    void wait(uint64_t msec)
    {
        std::chrono::milliseconds t(msec);
        std::unique_lock<std::mutex> locker(_mutex);
        _cond.wait_for(locker, t);
    }
    void signal() { _cond.notify_all(); }
    std::mutex &mutex() { return _mutex; }
};

} // namespace ds3d

#endif //
