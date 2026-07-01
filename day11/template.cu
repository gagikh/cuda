// Day 11: Textures and Surfaces
// Goal: bind a texture object and use it to zoom (upscale) an image with bilinear filtering.
//
// Compile:  nvcc -arch=sm_50 day11_template.cu -o day11
// Run:      ./day11

#include <cstdio>
#include <cuda_runtime.h>

// TODO 1: create a texture object bound to a plain device buffer (no OpenCV dependency).
// Use cudaResourceTypeLinear or cudaResourceTypePitch2D depending on your layout,
// cudaFilterModeLinear for bilinear filtering, cudaAddressModeClamp at the borders.
cudaTextureObject_t create_texture(const unsigned char *d_img, int width, int height, size_t pitch)
{
    cudaResourceDesc resDesc = {};
    // TODO: fill resDesc for a pitched 2D resource pointing at d_img

    cudaTextureDesc texDesc = {};
    // TODO: texDesc.filterMode = cudaFilterModeLinear;
    //       texDesc.addressMode[0] = texDesc.addressMode[1] = cudaAddressModeClamp;
    //       texDesc.readMode = cudaReadModeNormalizedFloat;
    //       texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex = 0;
    // TODO: cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
    return tex;
}

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

    cudaTextureObject_t tex = create_texture(d_img, width, height, pitch);

    unsigned char *d_out;
    cudaMalloc(&d_out, out_width * out_height);

    dim3 block(16, 16);
    dim3 grid((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y);
    zoom_kernel<<<grid, block>>>(tex, d_out, out_width, out_height, scale);
    cudaDeviceSynchronize();

    cudaDestroyTextureObject(tex);
    cudaFree(d_img);
    cudaFree(d_out);

    return 0;
}
