#version 450
// Test material cut out in stripes (an alpha tested material): its fragments discarded where fract(8 u) < 0.5 - in the shadow casters
// too (xlionrender_test_shadow_pairs.cpp: a masked caster through the GPU pairs against the CPU path). Else texture * 3, as the test material
#include "mb_standard_pbr.frag"

layout(binding = 1) uniform sampler2D SamplerAlbedo;

layout(location = 0) out vec4 outFragColor;

void main()
{
    if (fract(In.UV.x * 8.0) < 0.5) discard;
    outFragColor = vec4(texture(SamplerAlbedo, In.UV).rgb * 3.0, 1.0);
}
