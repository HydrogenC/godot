/**************************************************************************/
/*  bokeh_dof.cpp                                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "motion_blur.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

RendererRD::MotionBlur::MotionBlur(int p_tile_size) {
	tile_size = p_tile_size;

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

	// Hardcode tile size to enable unrolling of loops
	String tile_size_define = vformat("\n#define TILE_SIZE %d\n", tile_size);

	motion_blur.preprocess_shader.initialize({ "\n" });
	motion_blur.preprocess_shader_version = motion_blur.preprocess_shader.version_create();
	motion_blur.pipelines[MOTION_BLUR_PREPROCESS].create_compute_pipeline(motion_blur.preprocess_shader.version_get_shader(motion_blur.preprocess_shader_version, 0));

	motion_blur.tile_max_x_shader.initialize({ tile_size_define });
	motion_blur.tile_max_x_shader_version = motion_blur.tile_max_x_shader.version_create();
	motion_blur.pipelines[MOTION_BLUR_TILE_MAX_X].create_compute_pipeline(motion_blur.tile_max_x_shader.version_get_shader(motion_blur.tile_max_x_shader_version, 0));

	motion_blur.tile_max_y_shader.initialize({ tile_size_define });
	motion_blur.tile_max_y_shader_version = motion_blur.tile_max_y_shader.version_create();
	motion_blur.pipelines[MOTION_BLUR_TILE_MAX_Y].create_compute_pipeline(motion_blur.tile_max_y_shader.version_get_shader(motion_blur.tile_max_y_shader_version, 0));

	motion_blur.neighbor_max_shader.initialize({ "\n" });
	motion_blur.neighbor_max_shader_version = motion_blur.neighbor_max_shader.version_create();
	motion_blur.pipelines[MOTION_BLUR_NEIGHBOR_MAX].create_compute_pipeline(motion_blur.neighbor_max_shader.version_get_shader(motion_blur.neighbor_max_shader_version, 0));

	Vector<String> blur_modes;
	blur_modes.push_back("\n");
	blur_modes.push_back("\n#define USE_CUSTOM_CURVE\n");

	motion_blur.blur_shader.initialize(blur_modes);
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

void RendererRD::MotionBlur::motion_blur_process(RID base, RID depth, RID velocity, RID scene_data, RID custom_velocity, RID tile_max_x, RID tile_max_y, RID neighbor_max, RID output, Size2i base_size) {
	Size2i tiled_size = Size2i(Math::division_round_up(base_size.width, tile_size), Math::division_round_up(base_size.height, tile_size));

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	RD::get_singleton()->draw_command_begin_label("Preprocess motion vectors");

	RID shader = motion_blur.preprocess_shader.version_get_shader(motion_blur.preprocess_shader_version, 0);
	ERR_FAIL_COND(shader.is_null());

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, motion_blur.pipelines[MOTION_BLUR_PREPROCESS].get_rid());

	{
		RD::Uniform depth_texture_uniform = RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, {motion_blur.nearest_sampler, depth});
		RD::Uniform velocity_texture_uniform = RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, {motion_blur.nearest_sampler, velocity});
		RD::Uniform custom_velocity_image = RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, custom_velocity);
		RD::Uniform scene_data_uniform = RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 5, scene_data);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list,
			uniform_set_cache->get_cache(shader, 0, depth_texture_uniform, velocity_texture_uniform, custom_velocity_image, scene_data_uniform), 0);

		MotionBlurPreprocessPushConstant preprocess_push_constant;
		preprocess_push_constant.movement_velocity_multiplier = 1.0f;
		preprocess_push_constant.rotation_velocity_multiplier = 1.0f;
		preprocess_push_constant.object_velocity_multiplier = 1.0f;
		preprocess_push_constant.rotation_velocity_lower_threshold = 0.0f;
		preprocess_push_constant.rotation_velocity_upper_threshold = 0.0f;
		preprocess_push_constant.movement_velocity_lower_threshold = 0.0f;
		preprocess_push_constant.movement_velocity_upper_threshold = 0.0f;
		preprocess_push_constant.object_velocity_lower_threshold = 0.0f;
		preprocess_push_constant.object_velocity_upper_threshold = 0.0f;
		preprocess_push_constant.motion_blur_intensity = 5.0f;
		preprocess_push_constant.support_fsr2 = 1.0f;

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &preprocess_push_constant, sizeof(MotionBlurPreprocessPushConstant));
	}

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, base_size.x, base_size.y, 1);
	RD::get_singleton()->compute_list_add_barrier(compute_list);

	RD::get_singleton()->draw_command_end_label();
	RD::get_singleton()->draw_command_begin_label("Motion blur");

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, motion_blur.pipelines[MOTION_BLUR_TILE_MAX_X].get_rid());

	shader = motion_blur.tile_max_x_shader.version_get_shader(motion_blur.tile_max_x_shader_version, 0);
	ERR_FAIL_COND(shader.is_null());

	{
		RD::Uniform custom_velocity_uniform = RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, {motion_blur.nearest_sampler, custom_velocity});
		RD::Uniform depth_texture_uniform = RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, {motion_blur.nearest_sampler, depth});
		RD::Uniform tile_max_x_image = RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, tile_max_x);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list,
			uniform_set_cache->get_cache(shader, 0, custom_velocity_uniform, depth_texture_uniform, tile_max_x_image), 0);

		// Clear push constant
		RD::get_singleton()->compute_list_set_push_constant(compute_list, nullptr, 0);
	}

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, tiled_size.x, base_size.y, 1);
	RD::get_singleton()->compute_list_add_barrier(compute_list);

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, motion_blur.pipelines[MOTION_BLUR_TILE_MAX_Y].get_rid());

	shader = motion_blur.tile_max_y_shader.version_get_shader(motion_blur.tile_max_y_shader_version, 0);
	ERR_FAIL_COND(shader.is_null());

	{
		RD::Uniform tile_max_x_uniform = RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, {motion_blur.nearest_sampler, tile_max_x});
		RD::Uniform tile_max_y_image = RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, tile_max_y);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list,
			uniform_set_cache->get_cache(shader, 0, tile_max_x_uniform, tile_max_y_image), 0);
	}

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, tiled_size.x, tiled_size.y, 1);
	RD::get_singleton()->compute_list_add_barrier(compute_list);

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, motion_blur.pipelines[MOTION_BLUR_NEIGHBOR_MAX].get_rid());

	shader = motion_blur.neighbor_max_shader.version_get_shader(motion_blur.neighbor_max_shader_version, 0);
	ERR_FAIL_COND(shader.is_null());

	{
		RD::Uniform tile_max_y_uniform = RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, {motion_blur.nearest_sampler, tile_max_y});
		RD::Uniform neighbor_max_image = RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, neighbor_max);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list,
			uniform_set_cache->get_cache(shader, 0, tile_max_y_uniform, neighbor_max_image), 0);
	}

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, tiled_size.x, tiled_size.y, 1);
	RD::get_singleton()->compute_list_add_barrier(compute_list);

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, motion_blur.pipelines[MOTION_BLUR_BLUR].get_rid());

	shader = motion_blur.blur_shader.version_get_shader(motion_blur.blur_shader_version, 0);
	ERR_FAIL_COND(shader.is_null());

	{
		RD::Uniform color_texture_uniform = RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, {motion_blur.nearest_sampler, base});
		RD::Uniform custom_velocity_uniform = RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, {motion_blur.nearest_sampler, custom_velocity});
		RD::Uniform neighbor_max_uniform = RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 2, {motion_blur.nearest_sampler, neighbor_max});
		RD::Uniform output_image = RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, output);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list,
			uniform_set_cache->get_cache(shader, 0, color_texture_uniform, custom_velocity_uniform, neighbor_max_uniform, output_image), 0);

		MotionBlurBlurPushConstant blur_push_constant;
		blur_push_constant.motion_blur_intensity = 1.0f;
		blur_push_constant.sample_count = 16;
		blur_push_constant.frame = Engine::get_singleton()->get_frames_drawn() % 8;

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &blur_push_constant, sizeof(MotionBlurBlurPushConstant));
	}

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, base_size.x, base_size.y, 1);
	RD::get_singleton()->compute_list_add_barrier(compute_list);

	RD::get_singleton()->compute_list_end();
}

void RendererRD::MotionBlur::motion_blur_compute(Ref<RenderSceneBuffersRD> p_render_buffers, RenderSceneDataRD* p_scene_data, CopyEffects* p_copy_effects) {
	Size2i base_size = p_render_buffers->get_internal_size();
	uint32_t view_count = p_render_buffers->get_view_count();

	if (!p_render_buffers->has_texture(RB_SCOPE_MOTION_BLUR, RB_TEX_BLUR_OUTPUT)) {
		int usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
		Size2i tiled_size = Size2i(Math::division_round_up(base_size.width, tile_size), Math::division_round_up(base_size.height, tile_size));

		p_render_buffers->create_texture(RB_SCOPE_MOTION_BLUR, RB_TEX_CUSTOM_VELOCITY, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage_bits, RD::TEXTURE_SAMPLES_1, base_size);
		p_render_buffers->create_texture(RB_SCOPE_MOTION_BLUR, RB_TEX_TILE_MAX_X, RD::DATA_FORMAT_R16G16_SFLOAT, usage_bits, RD::TEXTURE_SAMPLES_1, Size2i(tiled_size.x, base_size.y));
		p_render_buffers->create_texture(RB_SCOPE_MOTION_BLUR, RB_TEX_TILE_MAX_Y, RD::DATA_FORMAT_R16G16_SFLOAT, usage_bits, RD::TEXTURE_SAMPLES_1, tiled_size);
		p_render_buffers->create_texture(RB_SCOPE_MOTION_BLUR, RB_TEX_NEIGHBOR_MAX, RD::DATA_FORMAT_R16G16_SFLOAT, usage_bits, RD::TEXTURE_SAMPLES_1, tiled_size);
		p_render_buffers->create_texture(RB_SCOPE_MOTION_BLUR, RB_TEX_BLUR_OUTPUT, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage_bits, RD::TEXTURE_SAMPLES_1, base_size);
	}

	RD::get_singleton()->draw_command_begin_label("Motion blur");
	for (uint32_t v = 0; v < view_count; v++) {
		RID base_texture = p_render_buffers->get_internal_texture(v);
		RID depth_texture = p_render_buffers->get_depth_texture(v);
		RID velocity_texture = p_render_buffers->get_velocity_buffer(false, v);

		RID custom_velocity = p_render_buffers->get_texture_slice(RB_SCOPE_MOTION_BLUR, RB_TEX_CUSTOM_VELOCITY, v, 0);
		RID tile_max_x = p_render_buffers->get_texture_slice(RB_SCOPE_MOTION_BLUR, RB_TEX_TILE_MAX_X, v, 0);
		RID tile_max_y = p_render_buffers->get_texture_slice(RB_SCOPE_MOTION_BLUR, RB_TEX_TILE_MAX_Y, v, 0);
		RID neighbor_max = p_render_buffers->get_texture_slice(RB_SCOPE_MOTION_BLUR, RB_TEX_NEIGHBOR_MAX, v, 0);
		RID output = p_render_buffers->get_texture_slice(RB_SCOPE_MOTION_BLUR, RB_TEX_BLUR_OUTPUT, v, 0);

		motion_blur_process(base_texture, depth_texture, velocity_texture, p_scene_data->get_uniform_buffer(), custom_velocity, tile_max_x, tile_max_y, neighbor_max, output, base_size);
		// Pong the blurred texture back to the internal texture
		p_copy_effects->copy_to_rect(output, base_texture, Rect2i(Point2i(), base_size));
	}


	RD::get_singleton()->draw_command_end_label();
}
