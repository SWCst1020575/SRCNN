#include "video.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}
#include <iostream>
#include <stdexcept>
void splitVideo(std::string f) {
    AVFormatContext* inctx;
    if (avformat_open_input(&inctx, f.c_str(), nullptr, nullptr) < 0) 
        throw "Failed to open target file.\n";
    
}