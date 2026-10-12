#version 450
// HDR -> exposure -> LionMapper (xLION's AgX derivation, what the xmaterial shaders use) -> sRGB encoded output.
#include "mb_tone_mapper_lion.frag"
#include "mb_lineartogamma.frag"

layout(binding = 0) uniform sampler2D SamplerHDR;

layout(push_constant) uniform Push
{
    vec4 Params;            // x = exposure
    vec4 Background;        // display-encoded color where nothing was drawn (alpha 0: transparent)
} push;

layout(location = 0) in  vec2 in_UV;
layout(location = 0) out vec4 outColor;

void main()
{
    const vec4 HDR = texture(SamplerHDR, in_UV);
    vec3 C   = HDR.rgb * push.Params.x;
    C        = clamp(ToneMapper_lion(max(C, vec3(1e-10))), 0.0, 1.0);
    const float Coverage = clamp(HDR.a, 0.0, 1.0);                  // 0 where nothing was drawn (cleared), 1 on geometry
    outColor = mix(push.Background, vec4(linearToSrgb(C), 1.0), Coverage);
}
