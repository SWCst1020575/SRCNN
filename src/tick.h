#ifndef __TICK_H__
#define __TICK_H__
#include <chrono>
namespace tick {
unsigned long getTickCount();
std::chrono::_V2::steady_clock::time_point getCurrent();
unsigned long getDiff(std::chrono::_V2::steady_clock::time_point start, std::chrono::_V2::steady_clock::time_point end);
};  // namespace tick

#endif  /// of __TICK_H__