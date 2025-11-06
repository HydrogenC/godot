#pragma once

#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/pipeline_deferred_rd.h"
#include "servers/rendering/renderer_rd/shaders/effects/motion_blur_blur.glsl.gen.h"

class MotionBlur {
public:
	enum MotionBlurMode {
		MOTION_BLUR_PREPROCESS,
		MOTION_BLUR_STANDARD,
		MOTION_BLUR_BOKEH,
		MOTION_BLUR_MAX,
	};

	struct {
		MotionBlurBlurShaderRD blur_shader;
		RID shader_version;
		PipelineDeferredRD compute_pipelines[MOTION_BLUR_MAX];
	} shaders;

};
