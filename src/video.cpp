#include "video.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}
#include <iostream>
#include <stdexcept>

#include "srcnn.h"
void* processVideo(void* file) {
    pthread_mutex_lock(&waitVideoMutex);
    av_register_all();
    AVFormatContext* inctx = nullptr;

    if (avformat_open_input(&inctx, (char*)file, nullptr, nullptr) < 0) {
        std::cerr << "Failed to open target file.\n";
        threadExit(-1);
    }
    if (avformat_find_stream_info(inctx, nullptr) < 0) {
        std::cerr << "fail to avformat_find_stream_info: ret=\n";
        threadExit(-1);
    }
    AVCodec* vcodec = nullptr;
    auto ret = av_find_best_stream(inctx, AVMEDIA_TYPE_VIDEO, -1, -1, &vcodec, 0);
    if (ret < 0) {
        std::cerr << "fail to av_find_best_stream: ret=" << ret;
        threadExit(-1);
    }
    const int vstrm_idx = ret;
    AVStream* vstrm = inctx->streams[vstrm_idx];

    // open video decoder context
    ret = avcodec_open2(vstrm->codec, vcodec, nullptr);
    if (ret < 0) {
        std::cerr << "fail to avcodec_open2: ret=" << ret;
        threadExit(-1);
    }

    const int dst_width = vstrm->codec->width;
    const int dst_height = vstrm->codec->height;
    setSrcSize(dst_height, dst_width);

    const AVPixelFormat dst_pix_fmt = AV_PIX_FMT_BGR24;
    SwsContext* swsctx = sws_getContext(
        vstrm->codec->width, vstrm->codec->height, AV_PIX_FMT_YUV420P,
        dst_width, dst_height, dst_pix_fmt, SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!swsctx) {
        std::cerr << "fail to sws_getCachedContext";
        threadExit(-1);
    }
    // allocate frame buffer for output
    AVFrame* frame = av_frame_alloc();
    std::vector<uint8_t> framebuf(avpicture_get_size(dst_pix_fmt, dst_width, dst_height));
    // av_image_fill_arrays(reinterpret_cast<AVPicture*>(frame), framebuf.data(), dst_pix_fmt, dst_width, dst_height);
    avpicture_fill(reinterpret_cast<AVPicture*>(frame), framebuf.data(), dst_pix_fmt, dst_width, dst_height);
    // decoding loop
    AVFrame* decframe = av_frame_alloc();
    unsigned nb_frames = 0;
    bool end_of_stream = false;
    int got_pic = 0;
    AVPacket pkt;
    av_init_packet(&pkt);
    do {
        if (!end_of_stream) {
            // read packet from input file
            ret = av_read_frame(inctx, &pkt);
            if (ret < 0 && ret != AVERROR_EOF) {
                std::cerr << "fail to av_read_frame: ret=" << ret;
                threadExit(-1);
            }
            if (ret == 0 && pkt.stream_index != vstrm_idx)
                goto next_packet;
            end_of_stream = (ret == AVERROR_EOF);
        }
        if (end_of_stream) {
            // null packet for bumping process
            av_init_packet(&pkt);
            pkt.data = nullptr;
            pkt.size = 0;
        }
        // decode video frame
        avcodec_decode_video2(vstrm->codec, decframe, &got_pic, &pkt);
        if (!got_pic)
            goto next_packet;
        // convert frame to OpenCV matrix
        sws_scale(swsctx, decframe->data, decframe->linesize, 0, decframe->height, frame->data, frame->linesize);

        // cv::Mat image(dst_height, dst_width, CV_8UC3, framebuf.data(), frame->linesize[0]);
        frameList.push_back(cv::Mat(dst_height, dst_width, CV_8UC3, frame->data[0], frame->linesize[0]).clone());
        pthread_mutex_unlock(&waitVideoMutex);
        // cv::imshow("press ESC to exit", image);
        // if (cv::waitKey(1) == 0x1b)
        //     break;

        // std::cout << nb_frames << '\r' << std::flush;  // dump progress
        // ++nb_frames;
    next_packet:
        av_free_packet(&pkt);
    } while (!end_of_stream || got_pic);
    av_frame_free(&decframe);
    av_frame_free(&frame);
    avcodec_close(vstrm->codec);
    avformat_close_input(&inctx);
    isVideoComplete = true;
}