#ifndef __VIDEO_H__
#define __VIDEO_H__
#include <list>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
extern void* processVideo(void*);
extern pthread_mutex_t waitVideoMutex;
extern std::list<cv::Mat> frameList;
extern std::vector<cv::Mat> frameListComplete;
extern bool isVideoComplete;
#endif