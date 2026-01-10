#include "motion_blur.h"

RendererRD::MotionBlur::MotionBlur() {
	Vector<String> motion_blur_modes;

	RD::SamplerState sampler;
	sampler.mag_filter = RD::SAMPLER_FILTER_NEAREST;
	sampler.min_filter = RD::SAMPLER_FILTER_NEAREST;
	sampler.mip_filter = RD::SAMPLER_FILTER_NEAREST;
	sampler.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
	sampler.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
	sampler.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
	motion_blur.nearest_sampler = RD::get_singleton()->sampler_create(sampler);

	sampler.mag_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler.min_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler.mip_filter = RD::SAMPLER_FILTER_LINEAR;
	motion_blur.linear_sampler = RD::get_singleton()->sampler_create(sampler);

	motion_blur.preprocess_shader.initialize({ "" });
	motion_blur.preprocess_shader_version = motion_blur.preprocess_shader.version_create();
	motion_blur.pipelines[MOTION_BLUR_PREPROCESS].create_compute_pipeline(motion_blur.preprocess_shader.version_get_shader(motion_blur.preprocess_shader_version, 0));

	motion_blur.tile_max_x_shader.initialize({ "" });
	motion_blur.tile_max_x_shader_version = motion_blur.tile_max_x_shader.version_create();
	motion_blur.pipelines[MOTION_BLUR_TILE_MAX_X].create_compute_pipeline(motion_blur.tile_max_x_shader.version_get_shader(motion_blur.tile_max_x_shader_version, 0));

	motion_blur.tile_max_y_shader.initialize({ "" });
	motion_blur.tile_max_y_shader_version = motion_blur.tile_max_y_shader.version_create();
	motion_blur.pipelines[MOTION_BLUR_TILE_MAX_Y].create_compute_pipeline(motion_blur.tile_max_y_shader.version_get_shader(motion_blur.tile_max_y_shader_version, 0));

	motion_blur.neighbor_max_shader.initialize({ "" });
	motion_blur.neighbor_max_shader_version = motion_blur.neighbor_max_shader.version_create();
	motion_blur.pipelines[MOTION_BLUR_NEIGHBOR_MAX].create_compute_pipeline(motion_blur.neighbor_max_shader.version_get_shader(motion_blur.neighbor_max_shader_version, 0));

	motion_blur.blur_shader.initialize({ "" });
	motion_blur.blur_shader_version = motion_blur.blur_shader.version_create();
	motion_blur.pipelines[MOTION_BLUR_BLUR].create_compute_pipeline(motion_blur.blur_shader.version_get_shader(motion_blur.blur_shader_version, 0));
}

RendererRD::MotionBlur::~MotionBlur() {
	for (int i = 0; i < MOTION_BLUR_MAX; i++) {
		motion_blur.pipelines[i].free();
	}

	motion_blur.preprocess_shader.version_free(motion_blur.preprocess_shader_version);
	motion_blur.tile_max_x_shader.version_free(motion_blur.tile_max_x_shader_version);
	motion_blur.tile_max_y_shader.version_free(motion_blur.tile_max_y_shader_version);
	motion_blur.neighbor_max_shader.version_free(motion_blur.neighbor_max_shader_version);
	motion_blur.blur_shader.version_free(motion_blur.blur_shader_version);

	RD::get_singleton()->free_rid(motion_blur.nearest_sampler);
	RD::get_singleton()->free_rid(motion_blur.linear_sampler);
}

void RendererRD::MotionBlur::allocate_buffers(Ref<RenderSceneBuffersRD> p_render_buffers, MotionBlurBuffers &p_buffers, const Size2i &p_size, int p_tile_size) {

}

void RendererRD::MotionBlur::motion_blur_compute(const MotionBlurBuffers &p_buffers) {

}
