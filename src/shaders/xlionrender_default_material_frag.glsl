#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

// The renderer's default material (instance::CreateDefaultMaterial): what a submesh whose material does not resolve draws with, on
// rigid and skinned geometry alike. Slot 0 the shadow map, slot 1 a diffuse texture; scene-linear HDR output (the renderer tone maps).
#include "mb_standard_pbr.frag"

layout(binding = 1) uniform sampler2D SamplerDiffuseMap;		// [INPUT_TEXTURE_DIFFUSE]

layout(location = 0) out vec4 outFragColor;

void main()
{
	vec3 FinalColor = PBRLighting
	( vec3(0, 0, 1)
	, texture(SamplerDiffuseMap, In.UV).rgb
	, 1
	, 0.04
	, 1
	, 0
	, vec3(0)
	);
	outFragColor = vec4(FinalColor, 1.0);
}
