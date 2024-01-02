#ifndef RECORD3DUNITYPLAYBACK_H
#define RECORD3DUNITYPLAYBACK_H

#include <cstdint>
#include <cstddef>

#if defined(WIN32) || defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {



EXPORT void DecompressFrame(unsigned char *jpgBytes,
                             uint32_t jpgBytesSize,
                             unsigned char *lzfseDepthBytes,
                             uint32_t lzfseBytesSize,
                             unsigned char *rgbBuffer,
                             float *poseBuffer,
                             int width, int height,
                             float fx, float fy, float tx, float ty);


EXPORT void DecompressColor(unsigned char *jpgBytes,
                             uint32_t jpgBytesSize,
                             unsigned char *rgbBuffer,
                             int &loadedRGBWidth,
                             int &loadedRGBHeight);

EXPORT size_t DecompressDepth(unsigned char* lzfseDepthBytes,
                     uint32_t lzfseBytesSize,
                     float** lzfseDecodedBytes,
                     int width, int height);

EXPORT void PopulatePositionBuffer(unsigned char *lzfseDecodedDepthBytes,
                    int loadedRGBWidth, int loadedRGBHeight,
                     uint32_t lzfseBytesSize,
                     float *poseBuffer,
                    size_t outSize,
                     int width, int height,
                     float fx, float fy, float tx, float ty);

EXPORT void TestFunction();

EXPORT void DecompressFrameDepthFast(unsigned char *lzfseDepthBytes,
                                     uint32_t lzfseBytesSize,
                                     float *poseBuffer,
                                     int width, int height,
                                     float fx, float fy, float tx, float ty);

EXPORT void DecompressFrameDepthReza(unsigned char *lzfseDepthBytes,
                                     uint32_t lzfseBytesSize,
                                     float *poseBuffer,
                                     int width, int height,
                                     float fx, float fy, float tx, float ty);
}





#endif // RECORD3DUNITYPLAYBACK_H
