# Day 11: Textures and Surfaces

## Objectives
- Create and bind CUDA texture objects
- Use `tex2D` fetches with filtering and addressing modes
- Implement image zoom and rotation via texture sampling

## Key Concepts
- Texture memory
- Surface memory
- Filtering & addressing
- Zoom/image processing

## Resources
https://developer.download.nvidia.com/CUDA/training/texture_webinar_aug_2011.pdf

http://cuda-programming.blogspot.com/2013/02/texture-memory-in-cuda-what-is-texture.html?m=1

## Code Walkthrough
Texture object wrapper around an OpenCV `GpuMat` (pitched 2D resource, linear filtering, clamp addressing):

```c++
template<class T>
struct filter_texture_t
{
	typedef T value_type;

	filter_texture_t(const cv::cuda::GpuMat& src) : m_src(src)
	{
		CV_Assert(sizeof(T) == src.elemSize());
		CV_Assert(0 == sizeof(T) % 4 || 1 == sizeof(T));

		cudaResourceDesc tex_resource;
		std::memset(&tex_resource, 0, sizeof(tex_resource));

		tex_resource.resType = cudaResourceTypePitch2D;
		tex_resource.res.pitch2D.devPtr = src.data;
		tex_resource.res.pitch2D.desc = cudaCreateChannelDesc<T>();
		tex_resource.res.pitch2D.width = src.cols;
		tex_resource.res.pitch2D.height = src.rows;
		tex_resource.res.pitch2D.pitchInBytes = src.step;

		cudaTextureDesc tex_descr;
		std::memset(&tex_descr, 0, sizeof(tex_descr));

		//tex_descr.readMode = cudaReadModeElementType;
		tex_descr.readMode = cudaReadModeNormalizedFloat;
		tex_descr.normalizedCoords = 0;
		tex_descr.addressMode[0] = cudaAddressModeClamp;
		tex_descr.addressMode[1] = cudaAddressModeClamp;
		const auto N = sizeof(tex_descr.borderColor) / sizeof(tex_descr.borderColor[0]);
		CV_Assert(4 == N);
		for (auto i = 0; i < N; ++i) {
			tex_descr.borderColor[i] = 0;
		}
		tex_descr.filterMode = cudaFilterModeLinear;
		//tex_descr.filterMode = cudaFilterModePoint;

		const auto err = cudaCreateTextureObject(&texture_, &tex_resource, &tex_descr, 0);
		if (cudaSuccess != err) {
			cv::error(err, "Texture creation error", __FUNCTION__, __FILE__, __LINE__);
		}
	}
	~filter_texture_t()
	{
		if (texture_ != 0)
		{
			const auto err = cudaDestroyTextureObject(texture_);
			if (cudaSuccess != err) {
				cv::error(err, "Texture destroy error", __FUNCTION__, __FILE__, __LINE__);
			}
			texture_ = 0;
		}
	}
// From the kernel
  tex2D<float4>(texture_, fx, fy);
```

## Hands-On Task
Zoom image, rotate.

## Self-Learning
1. Create a CUDA texture object bound to an image buffer (start from the wrapper above, or a plain array-based texture if you're not using OpenCV).
2. Implement image zoom (upscale) using `tex2D` bilinear filtering.
3. Implement image rotation by fetching from inverse-mapped coordinates through the texture.
4. Compare texture-based zoom against a manual shared-memory bilinear implementation (from Day 5) — which is simpler? Which is faster?

## Code Template
See [`template.cu`](template.cu) for a skeleton to start from.
