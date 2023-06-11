#ifndef __VIDEO_H__
#define __VIDEO_H__
#include <list>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
extern void* processVideo(void*);
extern void* combineVideo(void*);
extern pthread_mutex_t waitVideoMutex;
extern pthread_mutex_t videoCompleteMutex;
extern std::list<cv::Mat> frameList;
extern std::vector<cv::Mat> frameListComplete;
extern bool isVideoComplete;
extern unsigned nb_frames;
extern unsigned completeFrame;

#endif