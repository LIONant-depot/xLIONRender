#version 450
// HDR + bloom -> exposure -> LionMapper (xLION's AgX derivation, what the xmaterial shaders use) -> sRGB encoded output.
#include "mb_tone_mapper_lion.frag"
#include "mb_lineartogamma.frag"

layout(binding = 0) uniform sampler2D SamplerHDR;
layout(binding = 1) uniform sampler2D SamplerBloom;        // (xlionrender_post_comp.glsl)
layout(binding = 2) uniform sampler2D SamplerExposure;

layout(push_constant) uniform Push
{
    vec4 Params;            // x = exposure, y = the bloom's share, z = 1: auto exposure, w = 1: XLION_HEAT - binding 1 is the heat (xlionrender_heat_comp.glsl) over a grey image
    vec4 Background;        // display-encoded color where nothing was drawn (alpha 0: transparent)
} push;

layout(location = 0) in  vec2 in_UV;
layout(location = 0) out vec4 outColor;

void main()
{
    vec4 HDR = texture(SamplerHDR, in_UV);
    if (HDR.a > 1.0)
    {   // (the TAA's sharpening amount, above 1 in the alpha: this pixel against its cross, kept inside the cross's range - no halo)
        const vec3 N = textureOffset(SamplerHDR, in_UV, ivec2(0, -1)).rgb, S = textureOffset(SamplerHDR, in_UV, ivec2(0, 1)).rgb;
        const vec3 E = textureOffset(SamplerHDR, in_UV, ivec2(1, 0)).rgb,  W = textureOffset(SamplerHDR, in_UV, ivec2(-1, 0)).rgb;
        const vec3 Lo = min(min(min(N, S), min(E, W)), HDR.rgb), Hi = max(max(max(N, S), max(E, W)), HDR.rgb);
        HDR.rgb = clamp(HDR.rgb + (HDR.a - 1.0) * (HDR.rgb - 0.25 * (N + S + E + W)), Lo, Hi);
    }
    vec3 C   = (push.Params.y > 0.0 && push.Params.w == 0.0 ? HDR.rgb + texture(SamplerBloom, in_UV).rgb * push.Params.y : HDR.rgb) * push.Params.x;       // (added: what is under the threshold isn't in the bloom, a mix would dim it)
    if (push.Params.z > 0.0) C *= texelFetch(SamplerExposure, ivec2(0), 0).r;
    C        = clamp(ToneMapper_lion(max(C, vec3(1e-10))), 0.0, 1.0);
    const float Coverage = clamp(HDR.a, 0.0, 1.0);                  // 0 where nothing was drawn (cleared), 1 on geometry
    outColor = mix(push.Background, vec4(linearToSrgb(C), 1.0), Coverage);
    if (push.Params.w > 0.0)
    {
        const vec4 Heat = texture(SamplerBloom, in_UV);
        outColor = vec4(mix(vec3(dot(outColor.rgb, vec3(0.2126, 0.7152, 0.0722)) * 0.6), Heat.rgb, Heat.a * 0.85), 1.0);
    }
}
