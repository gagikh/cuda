// Day 11: Textures and Surfaces
// Goal: bind a texture object and use it to zoom (upscale) a real image with bilinear filtering.
//
// Compile:  nvcc -arch=sm_50 day11_template.cu -o day11 `pkg-config --cflags --libs opencv4`
// Run:      ./day11 <path-to-image>

#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudev.hpp>

// RAII texture wrapper, same shape as the filter_texture_t in the README's
// Code Walkthrough -- genericized to a raw pitched device pointer, so it
// works directly against a cv::cuda::GpuMat's .ptr()/.step (GpuMat rows are
// pitched, same idea as cudaMallocPitch from Day 4/5).
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
__global__ void zoom_kernel(cudaTextureObject_t tex, unsigned char *out, size_t out_step,
                             int out_width, int out_height, float scale)
{
    // TODO: int x = ..., y = ...;
    //       float fx = x / scale, fy = y / scale;
    //       float val = tex2D<float>(tex, fx + 0.5f, fy + 0.5f);
    //       out[y * out_step + x] = (unsigned char)(val * 255.0f);
}

// TODO 3 (self-learning #3): rotate_kernel — same idea, but map (x, y) through
// an inverse rotation matrix before sampling.

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: %s <path-to-image>\n", argv[0]);
        return 1;
    }

    cv::Mat h_img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (h_img.empty()) {
        printf("failed to load image: %s\n", argv[1]);
        return 1;
    }

    const float scale = 2.0f;
    const int out_width = (int)(h_img.cols * scale);
    const int out_height = (int)(h_img.rows * scale);

    cv::cuda::GpuMat d_img;
    d_img.upload(h_img);

    filter_texture_t<unsigned char> tex(d_img.ptr<unsigned char>(), d_img.cols, d_img.rows, d_img.step);

    cv::cuda::GpuMat d_out;
    d_out.create(out_height, out_width, d_img.type());

    dim3 block(16, 16);
    dim3 grid(cv::cudev::divUp(out_width, block.x), cv::cudev::divUp(out_height, block.y));
    zoom_kernel<<<grid, block>>>(tex, d_out.ptr<unsigned char>(), d_out.step, out_width, out_height, scale);
    cudaDeviceSynchronize();

    // tex's destructor calls cudaDestroyTextureObject automatically
    cv::Mat h_out;
    d_out.download(h_out);
    cv::imshow("input", h_img);
    cv::imshow("zoomed 2x", h_out);
    cv::waitKey(0);

    return 0;
}
