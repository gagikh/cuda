// Day 11: Textures and Surfaces
// Goal: bind a texture object and use it to zoom (upscale) an image with bilinear filtering.
//
// Compile:  nvcc -arch=sm_50 day11_template.cu -o day11
// Run:      ./day11

#include <cstdio>
#include <cuda_runtime.h>

// RAII texture wrapper, same shape as the filter_texture_t in the README's
// Code Walkthrough — genericized to a raw pitched device pointer instead of
// cv::cuda::GpuMat, so this file has no OpenCV dependency.
//
// TODO 1: fill in the constructor body: build a pitched-2D cudaResourceDesc
// pointing at d_ptr, a cudaTextureDesc with linear filtering + clamp
// addressing, then call cudaCreateTextureObject.
template<class T>
struct filter_texture_t
{
    using value_type = T;

    filter_texture_t(const T *d_ptr, int width, int height, size_t pitch)
    {
        cudaResourceDesc resDesc = {};
        // TODO: resDesc.resType = cudaResourceTypePitch2D;
        //       resDesc.res.pitch2D.devPtr = const_cast<T*>(d_ptr);
        //       resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
        //       resDesc.res.pitch2D.width = width;
        //       resDesc.res.pitch2D.height = height;
        //       resDesc.res.pitch2D.pitchInBytes = pitch;

        cudaTextureDesc texDesc = {};
        // TODO: texDesc.filterMode = cudaFilterModeLinear;
        //       texDesc.addressMode[0] = texDesc.addressMode[1] = cudaAddressModeClamp;
        //       texDesc.readMode = cudaReadModeNormalizedFloat;
        //       texDesc.normalizedCoords = 0;

        // TODO: const auto err = cudaCreateTextureObject(&texture_, &resDesc, &texDesc, nullptr);
        //       if (cudaSuccess != err) { /* handle error */ }
    }

    ~filter_texture_t()
    {
        if (texture_ != 0) {
            cudaDestroyTextureObject(texture_);
            texture_ = 0;
        }
    }

    // non-copyable (owns a GPU resource), movable if you need it later
    filter_texture_t(const filter_texture_t&) = delete;
    filter_texture_t& operator=(const filter_texture_t&) = delete;

    operator cudaTextureObject_t() const { return texture_; }

    cudaTextureObject_t texture_ = 0;
};

// TODO 2: zoom kernel — for each output pixel, map back to input coordinates
// (inverse scale) and sample via tex2D bilinear filtering.
__global__ void zoom_kernel(cudaTextureObject_t tex, unsigned char *out,
                             int out_width, int out_height, float scale)
{
    // TODO: int x = ..., y = ...;
    //       float fx = x / scale, fy = y / scale;
    //       float val = tex2D<float>(tex, fx + 0.5f, fy + 0.5f);
    //       out[y * out_width + x] = (unsigned char)(val * 255.0f);
}

// TODO 3 (self-learning #3): rotate_kernel — same idea, but map (x, y) through
// an inverse rotation matrix before sampling.

int main()
{
    const int width = 128, height = 128;
    const float scale = 2.0f;
    const int out_width = (int)(width * scale);
    const int out_height = (int)(height * scale);

    unsigned char *d_img;
    size_t pitch;
    cudaMallocPitch(&d_img, &pitch, width * sizeof(unsigned char), height);

    // TODO: fill d_img with test data

    filter_texture_t<unsigned char> tex(d_img, width, height, pitch);

    unsigned char *d_out;
    cudaMalloc(&d_out, out_width * out_height);

    dim3 block(16, 16);
    dim3 grid((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y);
    zoom_kernel<<<grid, block>>>(tex, d_out, out_width, out_height, scale);
    cudaDeviceSynchronize();

    // tex's destructor calls cudaDestroyTextureObject automatically
    cudaFree(d_img);
    cudaFree(d_out);

    return 0;
}
