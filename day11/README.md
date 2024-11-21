Tesxtures and surfaces
https://www.google.com/url?sa=t&source=web&rct=j&url=https://on-demand.gputechconf.com/gtc-express/2011/presentations/texture_webinar_aug_2011.pdf&ved=2ahUKEwjKoqCkvNH7AhUyuaQKHcIIDe8QFnoECCEQAQ&usg=AOvVaw1Cvy39n4v8bs4_pDp9PKbh

http://cuda-programming.blogspot.com/2013/02/texture-memory-in-cuda-what-is-texture.html?m=1

Zoom image


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
