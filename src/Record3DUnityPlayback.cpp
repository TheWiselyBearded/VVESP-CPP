#include "Record3DUnityPlayback.h"
#include <lzfse.h>
#include "JPEGDecoder.h"
#include <vector>
#include <thread>
#include <future>
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <iostream>
#include <omp.h>
// #include "/opt/homebrew/Cellar/libomp/17.0.5/include/omp.h"
#include <cstddef>


// #ifdef __x86_64__ || __i386__
// #ifdef defined(__x86_64__) || defined(__i386__)
#if defined __x86_64__
#include <emmintrin.h>
#endif

#ifdef __aarch64__  // Check for ARM64 architecture
#include <arm_neon.h>  // Include ARM NEON header for ARM64
#endif


static float InterpolateDepth(const float* $depthData, float $x, float $y, int $imgWidth, int $imgHeight)
{
    int wX = static_cast<int>($x);
    int wY = static_cast<int>($y);
    float fracX = $x - static_cast<float>(wX);
    float fracY = $y - static_cast<float>(wY);

    int topLeftIdx = wY * $imgWidth + wX;
    int topRightIdx = wY * $imgWidth + std::min(wX + 1, $imgWidth - 1);
    int bottomLeftIdx = std::min(wY + 1, $imgHeight - 1) * $imgWidth + wX;
    int bottomRightIdx = std::min(wY + 1, $imgHeight - 1) * $imgWidth + std::min(wX + 1, $imgWidth - 1);

    float interpVal =
            ($depthData[topLeftIdx]    * (1.0f - fracX) + fracX * $depthData[topRightIdx])    * (1.0f - fracY) +
            ($depthData[bottomLeftIdx] * (1.0f - fracX) + fracX * $depthData[bottomRightIdx]) * fracY;

    return interpVal;
}

#ifdef __aarch64__  // Check for ARM64 architecture
static float InterpolateDepthFast(const float* depthData, float x, float y, int imgWidth, int imgHeight)
{
    // Ensure x and y are within the bounds of the image
    x = std::max(0.0f, std::min(x, static_cast<float>(imgWidth - 1)));
    y = std::max(0.0f, std::min(y, static_cast<float>(imgHeight - 1)));

    int wX = static_cast<int>(x);
    int wY = static_cast<int>(y);
    float fracX = x - static_cast<float>(wX);
    float fracY = y - static_cast<float>(wY);

    // Calculate indices, ensuring they are within bounds
    int topLeftIdx = wY * imgWidth + wX;
    int topRightIdx = topLeftIdx + ((wX < imgWidth - 1) ? 1 : 0);
    int bottomLeftIdx = topLeftIdx + ((wY < imgHeight - 1) ? imgWidth : 0);
    int bottomRightIdx = topRightIdx + ((wY < imgHeight - 1) ? imgWidth : 0);

    // Load depth values safely
    float32x4_t topLeftVec = vld1q_dup_f32(&depthData[topLeftIdx]);
    float32x4_t topRightVec = vld1q_dup_f32(&depthData[topRightIdx]);
    float32x4_t bottomLeftVec = vld1q_dup_f32(&depthData[bottomLeftIdx]);
    float32x4_t bottomRightVec = vld1q_dup_f32(&depthData[bottomRightIdx]);

    float32x4_t fracXVec = vdupq_n_f32(fracX);
    float32x4_t fracYVec = vdupq_n_f32(fracY);

    // Perform vectorized interpolation
    float32x4_t resultVec = vaddq_f32(
        vmulq_f32(vsubq_f32(vdupq_n_f32(1.0f), fracXVec), topLeftVec),
        vmulq_f32(fracXVec, topRightVec)
    );
    resultVec = vaddq_f32(
        vmulq_f32(vsubq_f32(vdupq_n_f32(1.0f), fracYVec), resultVec),
        vmulq_f32(fracYVec, bottomLeftVec)
    );

    // Extract the result
    float result[4];
    vst1q_f32(result, resultVec);

    return result[0];
}
#endif


#if defined __x86_64__
static float InterpolateDepthFast(const float* depthData, float x, float y, int imgWidth, int imgHeight)
{
    int wX = static_cast<int>(x);
    int wY = static_cast<int>(y);
    float fracX = x - static_cast<float>(wX);
    float fracY = y - static_cast<float>(wY);

    int topLeftIdx = wY * imgWidth + wX;
    int topRightIdx = topLeftIdx + 1;
    int bottomLeftIdx = topLeftIdx + imgWidth;
    int bottomRightIdx = bottomLeftIdx + 1;

    // Perform bounds checking to avoid going out of bounds
    if (topRightIdx >= imgWidth)
        topRightIdx = topLeftIdx;
    if (bottomLeftIdx >= imgWidth * imgHeight)
        bottomLeftIdx = topLeftIdx;
    if (bottomRightIdx >= imgWidth * imgHeight)
        bottomRightIdx = topLeftIdx;

    __m128 topLeftVec = _mm_loadu_ps(&depthData[topLeftIdx]);
    __m128 topRightVec = _mm_loadu_ps(&depthData[topRightIdx]);
    __m128 bottomLeftVec = _mm_loadu_ps(&depthData[bottomLeftIdx]);
    __m128 bottomRightVec = _mm_loadu_ps(&depthData[bottomRightIdx]);

    __m128 fracXVec = _mm_set1_ps(fracX);
    __m128 fracYVec = _mm_set1_ps(fracY);

    // Perform vectorized interpolation
    __m128 resultVec = _mm_add_ps(
        _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(1.0f), fracXVec), topLeftVec),
        _mm_mul_ps(fracXVec, topRightVec)
    );
    resultVec = _mm_add_ps(
        _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(1.0f), fracYVec), resultVec),
        _mm_mul_ps(fracYVec, bottomLeftVec)
    );

    float result;
    _mm_store_ss(&result, resultVec);

    return result;
}
#endif


void DecompressFrame(unsigned char *jpgBytes,
                     uint32_t jpgBytesSize,
                     unsigned char *lzfseDepthBytes,
                     uint32_t lzfseBytesSize,

                     unsigned char *rgbBuffer,
                     float *poseBuffer,

                     int width, int height,
                     float fx, float fy, float tx, float ty)
{
    // 1. Decompress and copy the JPG bytes into the destination folder
    int loadedRGBWidth, loadedRGBHeight, loadedChannels;
    uint8_t* rgbPixels = stbi_load_from_memory( jpgBytes, jpgBytesSize, &loadedRGBWidth, &loadedRGBHeight, &loadedChannels, STBI_rgb );
    memcpy( rgbBuffer, rgbPixels, loadedRGBWidth * loadedRGBHeight * loadedChannels * sizeof(uint8_t));
    stbi_image_free( rgbPixels );

    // 2. Decompress the depth map
    size_t decompressedDepthMapSize = width * height * sizeof(float);
    auto* depthMap = (float*) malloc(decompressedDepthMapSize);
    size_t outSize = lzfse_decode_buffer((uint8_t*)depthMap, decompressedDepthMapSize, lzfseDepthBytes, lzfseBytesSize, nullptr);

    size_t depthmapDecompressedSizeIfSameResolutionAsRGB = loadedRGBWidth * loadedRGBHeight * sizeof(float);
    bool isDepthTheSameSizeAsRGB = depthmapDecompressedSizeIfSameResolutionAsRGB == outSize;

    int depthWidth = loadedRGBWidth;
    int depthHeight = loadedRGBHeight;

    // If the RGB image's resolution is different than the depth map resolution, then we are most probably working with
    // a higher-quality LiDAR video (RGB: 720x960 px, Depth: 192x256 px)
    if ( !isDepthTheSameSizeAsRGB )
    {
        depthWidth = 192;
        depthHeight = 256;
    }

    // Populate the pose buffer
    float ifx = 1.0f / fx;
    float ify = 1.0f / fy;
    float itx = -tx / fx;
    float ity = -ty / fy;

    float invRGBWidth = 1.0f / static_cast<float>(loadedRGBWidth);
    float invRGBHeight = 1.0f / static_cast<float>(loadedRGBHeight);

    constexpr int numComponentsPerPointPosition = 4;

    bool needToInterpolate = loadedRGBWidth != depthWidth || loadedRGBHeight != depthHeight;
    const float* depthDataPtr = (float*) depthMap;

    for (int i = 0; i < height; i++)
        for ( int j = 0; j < width; j++ )
        {
            int idx = loadedRGBWidth * i + j;
            int posBuffIdx = numComponentsPerPointPosition * idx;
            float depthX = invRGBWidth * depthWidth * j;
            float depthY = invRGBHeight * depthHeight * i;
            const float currDepth = needToInterpolate ? InterpolateDepth(depthDataPtr, depthX, depthY, depthWidth, depthHeight) : depthDataPtr[ idx ];

            poseBuffer[ posBuffIdx + 0 ] =  (ifx * j + itx) * currDepth;
            poseBuffer[ posBuffIdx + 1 ] = -(ify * i + ity) * currDepth;
            poseBuffer[ posBuffIdx + 2 ] = -currDepth;
            poseBuffer[ posBuffIdx + 3 ] = idx;
        }

    free(depthMap);
}

// Modify the function to accept two int parameters by reference
void DecompressColor(unsigned char *jpgBytes,
                     uint32_t jpgBytesSize,
                     unsigned char *rgbBuffer,
                     int &loadedRGBWidth,
                     int &loadedRGBHeight)
{
    int loadedChannels;
    uint8_t* rgbPixels = stbi_load_from_memory( jpgBytes, jpgBytesSize, &loadedRGBWidth, &loadedRGBHeight, &loadedChannels, STBI_rgb );
    memcpy( rgbBuffer, rgbPixels, loadedRGBWidth * loadedRGBHeight * loadedChannels * sizeof(uint8_t));
    stbi_image_free( rgbPixels );
}

size_t DecompressDepth(unsigned char* lzfseDepthBytes,
                     uint32_t lzfseBytesSize,
                     float** lzfseDecodedBytes,
                     int width, int height)
{
    // 2. Decompress the depth map
    size_t decompressedDepthMapSize = width * height * sizeof(float);

    // Ensure that the provided pointer points to a valid buffer
    if (*lzfseDecodedBytes == nullptr) {
        *lzfseDecodedBytes = (float*)malloc(decompressedDepthMapSize);
    } else {
        // Reallocate if the buffer size is different
        *lzfseDecodedBytes = (float*)realloc(*lzfseDecodedBytes, decompressedDepthMapSize);
    }

    size_t outSize = lzfse_decode_buffer((uint8_t*)*lzfseDecodedBytes, decompressedDepthMapSize, lzfseDepthBytes, lzfseBytesSize, nullptr);
    return outSize;
}

void PopulatePositionBuffer(unsigned char *lzfseDecodedDepthBytes,
                    int loadedRGBWidth, int loadedRGBHeight,
                     uint32_t lzfseBytesSize,
                     float *poseBuffer,
                    size_t outSize,
                     int width, int height,
                     float fx, float fy, float tx, float ty) {
                        
    size_t depthmapDecompressedSizeIfSameResolutionAsRGB = loadedRGBWidth * loadedRGBHeight * sizeof(float);
    bool isDepthTheSameSizeAsRGB = depthmapDecompressedSizeIfSameResolutionAsRGB == outSize;

    int depthWidth = loadedRGBWidth;
    int depthHeight = loadedRGBHeight;

    // If the RGB image's resolution is different than the depth map resolution, then we are most probably working with
    // a higher-quality LiDAR video (RGB: 720x960 px, Depth: 192x256 px)
    if ( !isDepthTheSameSizeAsRGB )
    {
        depthWidth = 192;
        depthHeight = 256;
    }

    // Populate the pose buffer
    float ifx = 1.0f / fx;
    float ify = 1.0f / fy;
    float itx = -tx / fx;
    float ity = -ty / fy;

    float invRGBWidth = 1.0f / static_cast<float>(loadedRGBWidth);
    float invRGBHeight = 1.0f / static_cast<float>(loadedRGBHeight);

    constexpr int numComponentsPerPointPosition = 4;

    bool needToInterpolate = loadedRGBWidth != depthWidth || loadedRGBHeight != depthHeight;
    const float* depthDataPtr = (float*) lzfseDecodedDepthBytes;

    for (int i = 0; i < height; i++)
        for ( int j = 0; j < width; j++ )
        {
            int idx = loadedRGBWidth * i + j;
            int posBuffIdx = numComponentsPerPointPosition * idx;
            float depthX = invRGBWidth * depthWidth * j;
            float depthY = invRGBHeight * depthHeight * i;
            const float currDepth = needToInterpolate ? InterpolateDepthFast(depthDataPtr, depthX, depthY, depthWidth, depthHeight) : depthDataPtr[ idx ];

            poseBuffer[ posBuffIdx + 0 ] =  (ifx * j + itx) * currDepth;
            poseBuffer[ posBuffIdx + 1 ] = -(ify * i + ity) * currDepth;
            poseBuffer[ posBuffIdx + 2 ] = -currDepth;
            poseBuffer[ posBuffIdx + 3 ] = idx;
        }

}

void TestFunction() {
    std::cout << "Test function called" << std::endl;
    // You can also just leave this empty for testing purposes
}

void DecompressFrameDepthReza(
                     unsigned char *lzfseDepthBytes,
                     uint32_t lzfseBytesSize,
                     float *poseBuffer,
                     int width, int height,
                     float fx, float fy, float tx, float ty)
{
    // 2. Decompress the depth map
    size_t decompressedDepthMapSize = width * height * sizeof(float);
    auto* depthMap = (float*) malloc(decompressedDepthMapSize);
    size_t outSize = lzfse_decode_buffer((uint8_t*)depthMap, decompressedDepthMapSize, lzfseDepthBytes, lzfseBytesSize, nullptr);

    size_t loadedRGBWidth = 720;
    size_t loadedRGBHeight = 960;
    size_t depthmapDecompressedSizeIfSameResolutionAsRGB = loadedRGBWidth * loadedRGBHeight * sizeof(float);
    bool isDepthTheSameSizeAsRGB = depthmapDecompressedSizeIfSameResolutionAsRGB == outSize;

    int depthWidth = loadedRGBWidth;
    int depthHeight = loadedRGBHeight;

    // If the RGB image's resolution is different than the depth map resolution, then we are most probably working with
    // a higher-quality LiDAR video (RGB: 720x960 px, Depth: 192x256 px)
    if ( !isDepthTheSameSizeAsRGB )
    {
        depthWidth = 192;
        depthHeight = 256;
    }

    // Populate the pose buffer
    float ifx = 1.0f / fx;
    float ify = 1.0f / fy;
    float itx = -tx / fx;
    float ity = -ty / fy;

    float invRGBWidth = 1.0f / static_cast<float>(loadedRGBWidth);
    float invRGBHeight = 1.0f / static_cast<float>(loadedRGBHeight);

    constexpr int numComponentsPerPointPosition = 4;

    bool needToInterpolate = loadedRGBWidth != depthWidth || loadedRGBHeight != depthHeight;
    const float* depthDataPtr = (float*) depthMap;

    for (int i = 0; i < height; i++)
        for ( int j = 0; j < width; j++ )
        {
            int idx = loadedRGBWidth * i + j;
            int posBuffIdx = numComponentsPerPointPosition * idx;
            float depthX = invRGBWidth * depthWidth * j;
            float depthY = invRGBHeight * depthHeight * i;
            const float currDepth = needToInterpolate ? InterpolateDepth(depthDataPtr, depthX, depthY, depthWidth, depthHeight) : depthDataPtr[ idx ];

            poseBuffer[ posBuffIdx + 0 ] =  (ifx * j + itx) * currDepth;
            poseBuffer[ posBuffIdx + 1 ] = -(ify * i + ity) * currDepth;
            poseBuffer[ posBuffIdx + 2 ] = -currDepth;
            poseBuffer[ posBuffIdx + 3 ] = idx;
        }

    free(depthMap);
}


void DecompressFrameDepthFast(
    unsigned char* lzfseDepthBytes,
    uint32_t lzfseBytesSize,
    float* poseBuffer,
    int width, int height,
    float fx, float fy, float tx, float ty)
{
    size_t decompressedDepthMapSize = width * height * sizeof(float);
    float* depthMap = new float[decompressedDepthMapSize];
    size_t outSize = lzfse_decode_buffer((uint8_t*)depthMap, decompressedDepthMapSize, lzfseDepthBytes, lzfseBytesSize, nullptr);
    // TODO: Fix dimensions
    int loadedRGBWidth = 720;
    int loadedRGBHeight = 960;
    int depthWidth = loadedRGBWidth;
    int depthHeight = loadedRGBHeight;

    if (decompressedDepthMapSize != (size_t)(loadedRGBWidth * loadedRGBHeight * sizeof(float))) {
        depthWidth = 192;
        depthHeight = 256;
    }

    float ifx = 1.0f / fx;
    float ify = 1.0f / fy;
    float itx = -tx / fx;
    float ity = -ty / fy;

    float invRGBWidth = 1.0f / static_cast<float>(loadedRGBWidth);
    float invRGBHeight = 1.0f / static_cast<float>(loadedRGBHeight);

    constexpr int numComponentsPerPointPosition = 4;

    bool needToInterpolate = (decompressedDepthMapSize != (size_t)(loadedRGBWidth * loadedRGBHeight * sizeof(float)));
    const float* depthDataPtr = depthMap;

    // Parallelize the outer loop using OpenMP
#pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = loadedRGBWidth * i + j;
            int posBuffIdx = numComponentsPerPointPosition * idx;
            float depthX = invRGBWidth * depthWidth * j;
            float depthY = invRGBHeight * depthHeight * i;
            const float currDepth = needToInterpolate ? InterpolateDepthFast(depthDataPtr, depthX, depthY, depthWidth, depthHeight) : depthDataPtr[idx];

            poseBuffer[posBuffIdx + 0] = (ifx * j + itx) * currDepth;
            poseBuffer[posBuffIdx + 1] = -(ify * i + ity) * currDepth;
            poseBuffer[posBuffIdx + 2] = -currDepth;
            poseBuffer[posBuffIdx + 3] = idx;
        }
    }

    delete[] depthMap;
}
