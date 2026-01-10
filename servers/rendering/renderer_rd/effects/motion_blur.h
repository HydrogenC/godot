#pragma once

#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/pipeline_deferred_rd.h"
#include "servers/rendering/renderer_rd/shaders/effects/motion_blur_blur.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/motion_blur_neighbor_max.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/motion_blur_preprocess.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/motion_blur_tile_max_x.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/motion_blur_tile_max_y.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"

#define RB_SCOPE_MOTION_BLUR SNAME("motion_blur")

#define RB_TEX_TILE_MAX_X SNAME("tile_max_x")
#define RB_TEX_TILE_MAX_Y SNAME("tile_max_y")
#define RB_TEX_NEIGHBOR_MAX SNAME("neighbor_max")
#define RB_TEX_BLUR_OUTPUT SNAME("blur_output")

namespace RendererRD {

class MotionBlur {
private:
	enum MotionBlurMode {
		MOTION_BLUR_PREPROCESS,
		MOTION_BLUR_TILE_MAX_X,
		MOTION_BLUR_TILE_MAX_Y,
		MOTION_BLUR_NEIGHBOR_MAX,
		MOTION_BLUR_BLUR,
		MOTION_BLUR_MAX,
	};

	struct MotionBlurPreprocessPushConstant {
		float rotation_velocity_multiplier;
		float movement_velocity_multiplier;
		float object_velocity_multiplier;
		float rotation_velocity_lower_threshold;

		float movement_velocity_lower_threshold;
		float object_velocity_lower_threshold;
		float rotation_velocity_upper_threshold;
		float movement_velocity_upper_threshold;

		float object_velocity_upper_threshold;
		float is_fsr2;
		float motion_blur_intensity;
		float pad;
	};

	struct MotionBlurTileMaxPushConstant {
		int32_t tile_size;
		int32_t pad[3];
	};

	struct  MotionBlurBlurPushConstant {
		float minimum_user_threshold;
		float importance_bias;
		float maximum_jitter_value;
		float motion_blur_intensity;

		int32_t tile_size;
		int32_t sample_count;
		int32_t frame;
		int32_t pad;
	};

	struct {
		MotionBlurPreprocessPushConstant preprocess_push_constant;
		MotionBlurTileMaxPushConstant tile_max_push_constant;
		MotionBlurBlurPushConstant blur_push_constant;

		MotionBlurPreprocessShaderRD preprocess_shader;
		RID preprocess_shader_version;

		MotionBlurTileMaxXShaderRD tile_max_x_shader;
		RID tile_max_x_shader_version;

		MotionBlurTileMaxYShaderRD tile_max_y_shader;
		RID tile_max_y_shader_version;

		MotionBlurNeighborMaxShaderRD neighbor_max_shader;
		RID neighbor_max_shader_version;

		MotionBlurBlurShaderRD blur_shader;
		RID blur_shader_version;

		PipelineDeferredRD pipelines[MOTION_BLUR_MAX];
		RID linear_sampler;
		RID nearest_sampler;
	} motion_blur;
public:
	struct MotionBlurBuffers {
		int tile_size;

		Size2i base_texture_size;
		// (base.x / tile_size, base.y)
		Size2i tile_x_texture_size;
		// (base.x / tile_size, base.y / tile_size)
		Size2i tile_y_texture_size;

		// textures
		RID base_texture;
		RID velocity_texture;
		RID tile_max_x_texture;
		RID tile_max_y_texture;
		RID neighbor_max_texture;
		RID output_texture;
	};

	MotionBlur();
	~MotionBlur();

	void allocate_buffers(Ref<RenderSceneBuffersRD> p_render_buffers, MotionBlurBuffers& p_buffers, const Size2i &p_size, int p_tile_size);
	void motion_blur_compute(const MotionBlurBuffers &p_buffers);
};
}
