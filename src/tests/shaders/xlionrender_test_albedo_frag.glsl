#version 450
// Test material: the standard PBR lighting with its albedo from a texture (slot 1) times the lab's wall red (0.75, 0.12, 0.10; a material
// draw's instance record does not carry the draw's constant color) - what the GI's capture
// and the reflections' hit shading read through its capture variant (xlionrender_test_reflections.cpp: the textured wall)
#include "mb_standard_pbr.frag"

struct InstanceData
{
    vec4   Rows[3];
    vec4   Material[3];    // [0] base color rgb, metallic; [1] emissive rgb, roughness; [2] x AO, y F0
    vec4   PrevRows[3];
};
layout(std430, set = 1, binding = 1) readonly buffer InstanceBuffer { InstanceData Instances[]; } instances;
layout(binding = 1) uniform sampler2D SamplerAlbedo;
layout(location = 8) flat in uint InInstance;
layout(location = 0) out vec4 outColor;

void main()
{
    const InstanceData I = instances.Instances[InInstance];
    outColor = vec4(PBRLighting(vec3(0.0, 0.0, 1.0), texture(SamplerAlbedo, In.UV).rgb * vec3(0.75, 0.12, 0.10), 1.0, 0.04, 0.8, 0.0, vec3(0.0)), 1.0);
}
