#ifndef _MSC_VER

#include "tick.h"

#include <sys/time.h>
#include <unistd.h>

#include <iostream>

namespace {
class __GET_TICK_COUNT {
   public:
    __GET_TICK_COUNT() {
        if (gettimeofday(&tv_, NULL) != 0)
            throw 0;
    }

    timeval tv_;
};

__GET_TICK_COUNT timeStart;
}  // namespace

namespace tick {

unsigned long getTickCount() {
    static time_t secStart = timeStart.tv_.tv_sec;
    static time_t usecStart = timeStart.tv_.tv_usec;
    timeval tv;
    gettimeofday(&tv, NULL);

    return (tv.tv_sec - secStart) * 1000 + (tv.tv_usec - usecStart) / 1000;
}
std::chrono::_V2::steady_clock::time_point getCurrent() {
    return std::chrono::steady_clock::now();
}
unsigned long getDiff(std::chrono::_V2::steady_clock::time_point start, std::chrono::_V2::steady_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}
};  // namespace tick

#endif  /// of _MSC_VER