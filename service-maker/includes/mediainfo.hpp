/**
 * @file
 * <b>MediaInfo class for aquiring media information</b>
 *
 *
 */
#ifndef NVIDIA_DEEPSTREAM_MEDIAINFO
#define NVIDIA_DEEPSTREAM_MEDIAINFO

#include <memory>
#include <string>
#include <vector>

namespace deepstream {

struct StreamInfo {
    std::string codec;

    virtual ~StreamInfo() {}
};

struct AudioStreamInfo : public StreamInfo {
    unsigned int channels;
};

struct VideoStreamInfo : public StreamInfo {
    unsigned int width;
    unsigned int height;
    struct {
        unsigned int num;
        unsigned int denom;
    } framerate;
};

struct MediaInfo {
    bool error = false;
    uint64_t duration = 0;
    bool live = false;
    operator bool() const { return !error; };
    std::vector<std::unique_ptr<StreamInfo>> streams;
    static std::unique_ptr<struct MediaInfo> discover(std::string uri);
};

} // namespace deepstream
#endif