///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// XeGTAO is based on GTAO/GTSO "Jimenez et al. / Practical Real-Time Strategies for Accurate Indirect Occlusion",
// https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
//
// Implementation:  Filip Strugar (filip.strugar@intel.com), Steve Mccalla <stephen.mccalla@intel.com>         (\_/)
// Version:         1.02                                                                                      (='.'=)
// Details:         https://github.com/GameTechDev/XeGTAO                                                     (")_(")
//
// Version history:
// 1.00 (2021-08-09): Initial release
// 1.01 (2021-09-02): Fix for depth going to inf for 'far' depth buffer values that are out of fp16 range
// 1.02 (2021-09-03): More fast_acos use and made final horizon cos clamping optional (off by default): 3-4% perf boost
// 1.10 (2021-09-03): Added a couple of heuristics to combat over-darkening errors in certain scenarios
// 1.20 (2021-09-06): Optional normal from depth generation is now a standalone pass: no longer integrated into
//                    main XeGTAO pass to reduce complexity and allow reuse; also quality of generated normals improved
// 1.21 (2021-09-28): Replaced 'groupshared'-based denoiser with a slightly slower multi-pass one where a 2-pass new
//                    equals 1-pass old. However, 1-pass new is faster than the 1-pass old and enough when TAA enabled.
// 1.22 (2021-09-28): Added 'XeGTAO_' prefix to all local functions to avoid name clashes with various user codebases.
// 1.30 (2021-10-10): Added support for directional component (bent normals).
// N/A  (2025-11-20): Port and convert to GLSL and Godot 4.6 by HydrogenC.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

#define GTAO_RADIUS_MULTIPLIER (0.5)
#define GTAO_MAX_DEPTH (1000.0)
#define GTAO_MAX_SCREEN_RADIUS (128.0)
#define GTAO_BIAS_MIP_LEVEL (0)
#define GTAO_FALLOFF_RANGE (0.717)
// The default intensity of ASSAO is 2.0, but XeGTAO uses 1.0 as base, so we scale it down to match the appearance.
#define GTAO_INTENSITY_SCALE (0.25)

#define PI 3.141592653589793
#define PI_HALF (PI / 2.0)
#define PI_INV 0.318309886183121

const int num_slices[5] = { 2, 4, 5, 6, 8 };
const int num_taps[5] = { 4, 6, 8, 12, 16 };

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	int quality;
	bool is_orthogonal;

	vec2 viewport_pixel_size;
	int pass;
	int pad;

	ivec2 pass_coord_offset;
	int size_multiplier;
	float fov_scale;

	vec2 NDC_to_view_mul;
	vec2 NDC_to_view_add;

	float radius;
	float intensity;
	float shadow_power;
	float shadow_clamp;

	float fade_out_mul;
	float fade_out_add;
	float inv_radius_near_limit;
	float thickness_heuristic;
}
params;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2DArray source_depth_mipmaps;

layout(rgba8, set = 0, binding = 1) uniform restrict readonly image2D source_normal;

layout(r8, set = 1, binding = 0) uniform restrict writeonly image2D dest_working_term;

layout(r8, set = 1, binding = 1) uniform restrict writeonly image2D dest_edges;

// packing/unpacking for edges; 2 bits per edge mean 4 gradient values (0, 0.33, 0.66, 1) for smoother transitions!
float pack_edges(vec4 p_edgesLRTB) {
	p_edgesLRTB = round(clamp(p_edgesLRTB, 0.0, 1.0) * 2.9);
	return dot(p_edgesLRTB, vec4(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));
}

vec4 calculate_edges(const float p_center_z, const float p_left_z, const float p_right_z, const float p_top_z, const float p_bottom_z) {
	vec4 edgesLRTB = vec4(p_left_z, p_right_z, p_top_z, p_bottom_z) - p_center_z;

	float slopeLR = (edgesLRTB.y - edgesLRTB.x) * 0.5;
	float slopeTB = (edgesLRTB.w - edgesLRTB.z) * 0.5;
	vec4 edgesLRTB_slope_adjusted = edgesLRTB + vec4(slopeLR, -slopeLR, slopeTB, -slopeTB);
	edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTB_slope_adjusted));
	return vec4(clamp((1.25 - edgesLRTB / (p_center_z * 0.011)), 0.0, 1.0));
}

vec3 NDC_to_viewspace(vec2 p_pos, float p_viewspace_depth) {
	if (params.is_orthogonal) {
		return vec3((params.NDC_to_view_mul * p_pos.xy + params.NDC_to_view_add), p_viewspace_depth);
	} else {
		return vec3((params.NDC_to_view_mul * p_pos.xy + params.NDC_to_view_add) * p_viewspace_depth, p_viewspace_depth);
	}
}

vec3 load_normal(ivec2 p_pos) {
	vec3 encoded_normal = normalize(imageLoad(source_normal, p_pos).xyz * 2.0 - 1.0);
	encoded_normal.z = -encoded_normal.z;
	return encoded_normal;
}

vec3 load_normal(ivec2 p_pos, ivec2 p_offset) {
	vec3 encoded_normal = normalize(imageLoad(source_normal, p_pos + p_offset).xyz * 2.0 - 1.0);
	encoded_normal.z = -encoded_normal.z;
	return encoded_normal;
}

// Calculate fadeout, intensity and power of AO value
float calculate_final_occlusion(float obscurance, float pix_center_z, vec4 edgesLRTB, int p_quality_level) {
	// calculate fadeout (1 close, gradient, 0 far)
	float fade_out = clamp(pix_center_z * params.fade_out_mul + params.fade_out_add, 0.0, 1.0);

	// Reduce the SSAO shadowing if we're on the edge to remove artifacts on edges (we don't care for the lower quality one)
	if (p_quality_level >= 2) {
		// when there's more than 2 opposite edges, start fading out the occlusion to reduce aliasing artifacts
		float edge_fadeout_factor = clamp((1.0 - edgesLRTB.x - edgesLRTB.y) * 0.35, 0.0, 1.0) + clamp((1.0 - edgesLRTB.z - edgesLRTB.w) * 0.35, 0.0, 1.0);

		fade_out *= clamp(1.0 - edge_fadeout_factor, 0.0, 1.0);
	}

	// strength
	obscurance = GTAO_INTENSITY_SCALE * params.intensity * obscurance;

	// clamp
	obscurance = min(obscurance, params.shadow_clamp);

	// fadeout
	obscurance *= fade_out;

	// conceptually switch to occlusion with the meaning being visibility (grows with visibility, occlusion == 1 implies full visibility),
	// to be in line with what is more commonly used.
	float occlusion = 1.0 - obscurance;

	// modify the gradient
	// note: this cannot be moved to a later pass because of loss of precision after storing in the render target
	occlusion = pow(clamp(occlusion, 0.0, 1.0), params.shadow_power);

	return occlusion;
}

// [Jimenez 2014] Interleaved gradient function
// Use integer coordinates rather than UV since UV varies too little
float quick_hash(vec2 pos) {
	const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
	return fract(magic.z * fract(dot(pos, magic.xy)));
}

vec2 get_angle_offset_noise(uvec2 coords) {
	coords.y = 4096u - coords.y;
	float angle = quick_hash(vec2(coords));
	// [Jorge Jiménez 2016] Practical Real-Time Strategies for Accurate Indirect Occlusion
	// Noise Distribution - Spatial Offsets
	float noise = 0.25 * float((coords.y - coords.x) & 3u);
	return vec2(angle, noise);
}

// [Eberly 2014] GPGPU Programming for Games and Science
// Fast approximation of arccos
float acos_fast(float x) {
	float abs_x = abs(x);
	float res = -0.156583 * abs_x + PI_HALF;
	res *= sqrt(1.0 - abs_x);
	return x >= 0.0 ? res : PI - res;
}

// GTAO with visibility bitmasks
#define BITMASK_SECTOR_COUNT 32

uint bit_count(uint value) {
	value = value - ((value >> 1u) & 0x55555555u);
	value = (value & 0x33333333u) + ((value >> 2u) & 0x33333333u);
	return ((value + (value >> 4u) & 0xF0F0F0Fu) * 0x1010101u) >> 24u;
}

// Refer to: https://cdrinmatane.github.io/posts/ssaovb-code/
uint update_sectors(float min_horizon, float max_horizon, uint global_occluded_bitmask)
{
	uint start_horizon_int = uint(min_horizon * BITMASK_SECTOR_COUNT);
	float angle_horizon = (max_horizon - min_horizon) * BITMASK_SECTOR_COUNT;
	uint angle_horizon_int = uint(ceil(angle_horizon));
	uint angle_horizon_bitmask = angle_horizon_int > 0 ? (0xFFFFFFFFu >> (BITMASK_SECTOR_COUNT - angle_horizon_int)) : 0u;
	uint current_occluded_bitmask = angle_horizon_bitmask << start_horizon_int;
	return global_occluded_bitmask | current_occluded_bitmask;
}

// Fast acos approximation (degree-3 polynomial)
vec2 fast_acos2(vec2 x)
{
	return ((-0.69813170 * x * x) - 0.87266463) * x + 1.57079633;
}

// Compute front and back horizon angles
vec2 get_front_back_horizons(float sampling_direction, vec3 delta_pos, vec3 view_vec, float n)
{
	sampling_direction = -sampling_direction; // Flip to align with left-handed view space and Y-down
	vec3 delta_pos_backface = delta_pos - view_vec * params.thickness_heuristic;

	vec2 front_back_horizon = vec2(
	dot(normalize(delta_pos), view_vec),
	dot(normalize(delta_pos_backface), view_vec)
	);

	front_back_horizon = fast_acos2(front_back_horizon);

	front_back_horizon = clamp(
	(sampling_direction * -front_back_horizon - n + PI_HALF) / PI,
	0.0, 1.0
	);

	// Conditional swizzle: if sampling_direction >= 0.0, return .yx, else .xy
	front_back_horizon = sampling_direction >= 0.0
	? front_back_horizon.yx
	: front_back_horizon.xy;

	return front_back_horizon;
}

float GTAO_slice(in int num_slices, in int num_taps, in int slice_index, float sample_noise, vec2 base_uv, vec2 screen_dir, float search_radius, vec3 view_pos, vec3 view_dir, vec3 view_space_normal) {
	float scene_depth, sample_delta_len_sq, sample_horizon_cos, falloff;
	vec3 sample_delta;
	vec2 sample_uv;
	const vec2 screen_vec_uv = screen_dir * params.viewport_pixel_size;
	const float thickness = params.thickness_heuristic;

	// Project view_space_normal onto the plane defined by screen_dir and view_dir
	vec3 axis_vec = normalize(cross(view_dir, vec3(screen_dir, 0.0)));
	vec3 ortho_dir_vec = cross(view_dir, axis_vec);
	vec3 proj_normal_vec = view_space_normal - axis_vec * dot(view_space_normal, axis_vec);

	float proj_normal_len = length(proj_normal_vec) + 0.000001;

	float sign_norm = sign(dot(ortho_dir_vec, proj_normal_vec));
	float cos_norm = dot(proj_normal_vec, view_dir) / proj_normal_len;
	float n = sign_norm * acos_fast(cos_norm);

	uint occlusion_bitmask = 0u;

	// Find the largest angle
	for (int i = 0; i < num_taps; ++i) {
		float step_noise = float(slice_index + i * num_taps) * 0.6180339887498948482; // <- this should unroll
		step_noise = fract(sample_noise + step_noise);

		float s = (i + step_noise) / num_taps;
		s *= s;

		vec2 uv_offset = screen_vec_uv * max(search_radius * s, float(i) + 1.0);
		// Paper: flip y due to texture coordinate system
		uv_offset.y *= -1.0;

		// Use HZB tracing for better performance
		int mip_level = GTAO_BIAS_MIP_LEVEL;
		if (i == 2) {
			mip_level++;
		}

		if (i >= 3) {
			mip_level += 2;
		}

		float scene_depth;
		vec3 delta_pos1, delta_pos2;
		vec2 sample_uv, sample_horizon_cos, delta_pos_len_sq;

		// Positive direction
		// Clamp UV coords to avoid artifacts
		sample_uv = base_uv + uv_offset;
		scene_depth = textureLod(source_depth_mipmaps, vec3(sample_uv, params.pass), mip_level).x;
		delta_pos1 = NDC_to_viewspace(sample_uv, scene_depth).xyz - view_pos;
		delta_pos_len_sq.x = dot(delta_pos1, delta_pos1);
		// TODO: This could be replaced with fast rsqrt
		sample_horizon_cos.x = dot(delta_pos1, view_dir) * inversesqrt(delta_pos_len_sq.x);

		// Negative direction
		sample_uv = base_uv - uv_offset;
		scene_depth = textureLod(source_depth_mipmaps, vec3(sample_uv, params.pass), mip_level).x;
		delta_pos2 = NDC_to_viewspace(sample_uv, scene_depth).xyz - view_pos;
		delta_pos_len_sq.y = dot(delta_pos2, delta_pos2);
		sample_horizon_cos.y = dot(delta_pos2, view_dir) * inversesqrt(delta_pos_len_sq.y);

		float back_horizon;
		vec2 front_back_horizon;

		// Update visibility mask for both directions
		back_horizon = dot(normalize(delta_pos1 - view_dir * thickness), view_dir);
		front_back_horizon.x = clamp((-acos(sample_horizon_cos.x) - n + PI_HALF) * PI_INV, 0.0, 1.0);
		front_back_horizon.y = clamp((-acos(back_horizon) - n + PI_HALF) * PI_INV, 0.0, 1.0);

		// `front_back_horizon.x` is larger than `.y`
		occlusion_bitmask = update_sectors(front_back_horizon.y, front_back_horizon.x, occlusion_bitmask);

		back_horizon = dot(normalize(delta_pos2 - view_dir * thickness), view_dir);
		front_back_horizon.x = clamp((acos_fast(sample_horizon_cos.y) - n + PI_HALF) * PI_INV, 0.0, 1.0);
		front_back_horizon.y = clamp((acos_fast(back_horizon) - n + PI_HALF) * PI_INV, 0.0, 1.0);

		// `front_back_horizon.y` is larger than `.x`
		occlusion_bitmask = update_sectors(front_back_horizon.x, front_back_horizon.y, occlusion_bitmask);
	}

	float local_visibility = 1.0 - float(bit_count(occlusion_bitmask)) / float(BITMASK_SECTOR_COUNT);
	// Disallow total occlusion
	local_visibility = max(0.03, local_visibility);
	return local_visibility;
}

void generate_GTAO_shadows_internal(out float r_shadow_term, out vec4 r_edges, out float r_weight, const ivec2 p_pix_coord, int p_quality_level) {
	vec2 normalized_screen_pos = (p_pix_coord + vec2(0.5)) * params.viewport_pixel_size;

	// Load this pixel's viewspace normal
	uvec2 full_res_coord = p_pix_coord * params.size_multiplier + params.pass_coord_offset.xy;
	vec3 pixel_normal = load_normal(ivec2(full_res_coord));

	const int number_of_taps = num_taps[p_quality_level];
	const int number_of_slices = num_slices[p_quality_level];
	float pix_z, pix_left_z, pix_top_z, pix_right_z, pix_bottom_z;

	vec3 gather_pos = vec3(vec2(p_pix_coord) * params.viewport_pixel_size, params.pass);
	vec4 valuesUL = textureGather(source_depth_mipmaps, gather_pos);
	vec4 valuesBR = textureGatherOffset(source_depth_mipmaps, gather_pos, ivec2(1, 1));

	// get this pixel's viewspace depth
	pix_z = valuesUL.y;

	// get left right top bottom neighboring pixels for edge detection (gets compiled out on quality_level == 0)
	pix_left_z = valuesUL.x;
	pix_top_z = valuesUL.z;
	pix_right_z = valuesBR.z;
	pix_bottom_z = valuesBR.x;

	// Calculate edges
	vec4 edgesLRTB = calculate_edges(pix_z, pix_left_z, pix_right_z, pix_top_z, pix_bottom_z);

	// Move center pixel slightly towards camera to avoid imprecision artifacts due to depth buffer imprecision
	pix_z *= 0.99920;

	vec3 pix_center_pos = NDC_to_viewspace(normalized_screen_pos, pix_z);
	vec3 view_dir = normalize(-pix_center_pos);

	if (pix_z >= GTAO_MAX_DEPTH) {
		// Skip GTAO calculation if pixel is too far away
		r_shadow_term = 1.0;
		r_edges = edgesLRTB;
		r_weight = 1.0;
		return;
	}

	// Calculate rotation angle for slices
	float delta_angle = PI / float(number_of_slices);
	// Precalculate rotational components for slices
	float sin_delta_angle = sin(delta_angle);
	float cos_delta_angle = cos(delta_angle);

	float viewspace_radius = params.radius * GTAO_RADIUS_MULTIPLIER;

	// when too close, on-screen sampling disk will grow beyond screen size; limit this to avoid closeup temporal artifacts
	const float too_close_limit = clamp(length(pix_center_pos) * params.inv_radius_near_limit, 0.0, 1.0) * 0.8 + 0.2;

	viewspace_radius *= too_close_limit;

	// Multiply the radius by projection[0][0] to make it FOV-independent, same as HBAO
	// float screenspace_radius = clamp(viewspace_radius * params.fov_scale / pix_center_pos.z, float(number_of_taps), GTAO_MAX_SCREEN_RADIUS);

	// Adjust radius near screen borders to reduce artifacts
	// float near_screen_border = min(min(normalized_screen_pos.x, 1.0 - normalized_screen_pos.x), min(normalized_screen_pos.y, 1.0 - normalized_screen_pos.y));
	// near_screen_border = clamp(10.0 * near_screen_border + 0.6, 0.0, 1.0);
	// screenspace_radius *= near_screen_border;

	float pixel_dir_rb_viewspace_size_at_center_z = pix_z * params.viewport_pixel_size.x * params.NDC_to_view_mul.x;
	float screenspace_radius = params.radius / pixel_dir_rb_viewspace_size_at_center_z;

	vec2 noise = get_angle_offset_noise(p_pix_coord);
	// Get a random direction on the hemisphere
	// Screen dir is guaranteed to be already normalized
	vec2 screen_dir;
	screen_dir.y = sin(noise.x);
	screen_dir.x = cos(noise.x);

	// the main obscurance & sample weight storage
	float obscurance_sum = 0.0;
	float weight_sum = 0.0;

	// Calculate AO values for each slice
	for (int slice = 0; slice < number_of_slices; ++slice) {
		// GTAO inner integral gives visibility, which is one minus obscurance
		obscurance_sum += 1.0 - GTAO_slice(number_of_slices, number_of_taps, slice, noise.y, normalized_screen_pos, screen_dir, screenspace_radius, pix_center_pos, view_dir, pixel_normal);
		weight_sum += 1.0;

		// XeGTAO calculates screen direction with sincos(angle) every iteration, but that's too slow,
		// so we calculate it once and rotate it instead.
		vec2 tmp_dir = screen_dir;
		screen_dir.x = tmp_dir.x * cos_delta_angle - tmp_dir.y * sin_delta_angle;
		screen_dir.y = tmp_dir.x * sin_delta_angle + tmp_dir.y * cos_delta_angle;
	}

	// calculate weighted average
	float obscurance = obscurance_sum / weight_sum;

	// calculate final occlusion
	float occlusion = calculate_final_occlusion(obscurance, pix_center_pos.z, edgesLRTB, p_quality_level);

	// outputs!
	r_shadow_term = occlusion; // Our final 'occlusion' term (0 means fully occluded, 1 means fully lit)
	r_edges = edgesLRTB; // These are used to prevent blurring across edges, 1 means no edge, 0 means edge, 0.5 means half way there, etc.
	r_weight = weight_sum;
}

void main() {
	float out_shadow_term;
	float out_weight;
	vec4 out_edges;
	ivec2 pix_coord = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThanEqual(pix_coord, params.screen_size))) { //too large, do nothing
		return;
	}

	generate_GTAO_shadows_internal(out_shadow_term, out_edges, out_weight, pix_coord, params.quality);
	if (params.quality == 0) {
		out_edges = vec4(1.0);
	}

	imageStore(dest_working_term, ivec2(gl_GlobalInvocationID.xy), vec4(out_shadow_term, 0.0, 0.0, 0.0));
	imageStore(dest_edges, ivec2(gl_GlobalInvocationID.xy), vec4(pack_edges(out_edges), 0.0, 0.0, 0.0));
}
