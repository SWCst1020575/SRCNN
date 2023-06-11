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
AVCodecContext* c;
static void encode(AVCodecContext*, AVFrame*, AVPacket*, FILE*);
void* processVideo(void* file) {
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
    c = avcodec_alloc_context3(vcodec);
    avcodec_copy_context(c, vstrm->codec);
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
    nb_frames = 0;
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
        ++nb_frames;
    next_packet:
        av_free_packet(&pkt);
    } while (!end_of_stream || got_pic);
    av_frame_free(&decframe);
    av_frame_free(&frame);
    avcodec_close(vstrm->codec);
    avformat_close_input(&inctx);
    isVideoComplete = true;
}
void* combineVideo(void* file) {
    pthread_mutex_lock(&videoCompleteMutex);
    av_log_set_level(AV_LOG_FATAL);
    uint8_t endcode[] = {0, 0, 1, 0xb7};
    FILE* f;
    f = fopen((char*)file, "wb");
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        std::cerr << "Error finding H.264 encoder" << std::endl;
        threadExit(-1);
    }
    
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    codec_ctx->codec_id = c->codec_id;
    codec_ctx->codec_type = c->codec_type;
    codec_ctx->width = frameListComplete[0].cols;
    codec_ctx->height = frameListComplete[0].rows;
    codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx->bit_rate = c->bit_rate;
    codec_ctx->time_base = (AVRational){c->framerate.den, c->framerate.num};
    codec_ctx->framerate = (AVRational){c->framerate.num, c->framerate.den};
    codec_ctx->codec_tag = c->codec_tag;
    codec_ctx->gop_size = c->gop_size;
    codec_ctx->max_b_frames = c->max_b_frames;
    codec_ctx->flags = c->flags;
    // stream->time_base = c->time_base;
    AVDictionary* opt = NULL;

    if (avcodec_open2(codec_ctx, codec, &opt) < 0) {
        std::cerr << "Error opening codec." << std::endl;
        avcodec_free_context(&codec_ctx);
        threadExit(-1);
    }
    AVFrame* frame = av_frame_alloc();
    av_image_alloc(frame->data, frame->linesize, codec_ctx->width, codec_ctx->height,
                   AVPixelFormat::AV_PIX_FMT_YUV420P, 1);
    SwsContext* conversion = sws_getContext(
        codec_ctx->width, codec_ctx->height, AVPixelFormat::AV_PIX_FMT_BGR24,  // codec_ctx->pix_fmt,  <--- The source format comes from the input AVFrame
        codec_ctx->width, codec_ctx->height, AV_PIX_FMT_YUV420P, SWS_BICUBIC, nullptr, nullptr, nullptr);
    frame->format = AV_PIX_FMT_YUV420P;
    frame->width = codec_ctx->width;
    frame->height = codec_ctx->height;
    if (av_frame_get_buffer(frame, 0) < 0) {
        std::cerr << "Could not allocate the video frame data\n";
        avcodec_free_context(&codec_ctx);
        av_frame_free(&frame);
        threadExit(-1);
    }

    AVPacket* pkt = av_packet_alloc();
    for (completeFrame = 0; completeFrame < nb_frames; completeFrame++) {
        while (completeFrame == frameListComplete.size()) {
        }
        int cvLinesizes[1];
        cvLinesizes[0] = frameListComplete[completeFrame].step1();
       

        sws_scale(conversion, &frameListComplete[completeFrame].data, cvLinesizes, 0, frameListComplete[completeFrame].rows, frame->data,
                  frame->linesize);
        frame->pts = completeFrame + 1;
        encode(codec_ctx, frame, pkt, f);
    }
    if (codec->id == AV_CODEC_ID_MPEG1VIDEO || codec->id == AV_CODEC_ID_MPEG2VIDEO)
        fwrite(endcode, 1, sizeof(endcode), f);
    fclose(f);
    av_packet_free(&pkt);
    sws_freeContext(conversion);
    av_frame_free(&frame);
    avcodec_free_context(&c);
    avcodec_free_context(&codec_ctx);
}
static void encode(AVCodecContext* enc_ctx, AVFrame* frame, AVPacket* pkt, FILE* outfile) {
    int ret;

    /* send the frame to the encoder */
    // if (frame)
    //     printf("Send frame %3" PRId64 "\n", frame->pts);

    ret = avcodec_send_frame(enc_ctx, frame);
    if (ret < 0) {
        std::cerr << "Error sending a frame for encoding.\n";
        // fprintf(stderr, "Error sending a frame for encoding\n");
        exit(1);
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            fprintf(stderr, "Error during encoding\n");
            exit(1);
        }
        // printf("Write packet %3" PRId64 " (size=%5d)\n", pkt->pts, pkt->size);
        fwrite(pkt->data, 1, pkt->size, outfile);
        av_packet_unref(pkt);
    }
}