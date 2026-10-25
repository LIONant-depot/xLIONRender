#version 450
// The GI at half resolution (instance::renderGIScreen): the scene's rigid GI draws into a half-size depth, and each visible pixel runs
// the material's own GI read (mb_standard_pbr.frag GIPixelGI: the pixel read of the surfels, the vertex read, the local mean, the read
// toward the reflection) with the surface's own normal (no normal map), into a buffer the main pass upsamples by depth and normal
// (xlionrender_surfel_gi.glsl SurfelGIScreen). Its entry 0: the half size; then a uvec4 a pixel - the irradiance (half RGB), the
// distance to the eye (half), the reflection's (half RGB), the normal (7 + 7 bits), bit 30: irradiance read, bit 31: reflection read
#include "mb_standard_pbr.frag"

layout(early_fragment_tests) in;             // (only the nearest surface writes: the depth was laid first, this pass tests equal)
layout(std430, set = 1, binding = 11) buffer GIScreenBuffer { uvec4 GIScreen[]; };

void main()
{
    const vec3 N = TransformNotmalToWorld(vec3(0.0, 0.0, 1.0));
    const vec3 V = normalize(UBOLight.wSpaceEyePos.xyz - In.wSpacePosition.xyz);
    GIPixelGI(vec3(0.0, 0.0, 1.0), N, V, vec3(1.0), 0.0, vec3(0.0));
    const uint  W = GIScreen[0].x, H = GIScreen[0].y;
    const ivec2 P = ivec2(gl_FragCoord.xy);
    if (uint(P.x) >= W || uint(P.y) >= H) return;
    const bool  bE = g_GI.w > 0.5, bR = g_GIRefl.w > 0.5;
    const vec3  E = bE ? g_GI.rgb : vec3(0.0), R = bR ? g_GIRefl.rgb : vec3(0.0);
    vec2 o = N.xy / (abs(N.x) + abs(N.y) + abs(N.z));
    if (N.z < 0.0) o = (1.0 - abs(o.yx)) * vec2(o.x >= 0.0 ? 1.0 : -1.0, o.y >= 0.0 ? 1.0 : -1.0);
    const uvec2 Oq = uvec2(clamp(o * 0.5 + 0.5, 0.0, 1.0) * 127.0 + 0.5);
    GIScreen[1u + uint(P.y) * W + uint(P.x)] = uvec4(packHalf2x16(E.rg), packHalf2x16(vec2(E.b, length(In.wSpacePosition.xyz))), packHalf2x16(R.rg),
                                                    (packHalf2x16(vec2(R.b, 0.0)) & 0xFFFFu) | (Oq.x << 16) | (Oq.y << 23) | (bE ? 1u << 30 : 0u) | (bR ? 1u << 31 : 0u));
}
