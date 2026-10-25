#version 450
// The GI at half resolution (instance::renderGIScreen): the scene's rigid GI draws into a half-size depth, and each visible pixel runs
// the material's own GI read (mb_standard_pbr.frag GIPixelGI: the pixel read of the surfels, the vertex read, the local mean, the read
// toward the reflection) with the surface's own normal (no normal map), into a buffer the main pass upsamples by depth and normal
// (mb_standard_pbr.frag SurfelGIScreen). Its header (8 entries): 0 the half size, the frame, bit 0 last frame's buffer is there + bits
// 1-2 the fresh rate (log2 of 1 in N tiles); 1-4 last frame's camera-centered world to clip (columns); 5 xyz this frame's eye minus last
// frame's; 6 the counters (fresh by the pattern, reprojected, rejected then fresh, too old then fresh). Then a uvec4 a pixel - the
// irradiance (half RGB), the distance to the eye (half), the reflection's (half RGB), the normal (6 + 6 bits, bits 16-27), its age
// (bits 28-29: frames since a read made it), bit 30: irradiance read, bit 31: reflection read.
// Temporal: the 8x8 tiles of the frame's pattern (1 in N, rotating: a warp all fresh or all reprojected) run the read; the others take
// last frame's result where this point was (reprojected) if it is the same surface there (its distance and normal) and younger than
// screen_max_age_v, and run the read where it is not
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#include "mb_standard_pbr.frag"

layout(early_fragment_tests) in;             // (only the nearest surface writes: the depth was laid first, this pass tests equal)
layout(std430, set = 1, binding = 11) buffer GIScreenBuffer { uvec4 GIScreen[]; };
layout(std430, set = 1, binding = 13) readonly buffer GIScreenPrevBuffer { uvec4 GIScreenPrev[]; };

const uint screen_header_v  = 8u;
const uint screen_max_age_v = 3u;

vec3 ScreenNormal(in const uint w)
{
    const vec2 o = vec2(float((w >> 16) & 63u), float((w >> 22) & 63u)) / 63.0 * 2.0 - 1.0;
    vec3 n = vec3(o, 1.0 - abs(o.x) - abs(o.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

void main()
{
    g_GIFaceD = cross(dFdx(InGIPos), dFdy(InGIPos)); g_bGIFaceD = true;      // (before any lane leaves the quad's work: the read's face normal)
    const uvec4 Hd = GIScreen[0];
    const uint  W = Hd.x, H = Hd.y;
    const ivec2 P = ivec2(gl_FragCoord.xy);
    const bool  bIn = uint(P.x) < W && uint(P.y) < H;
    const vec3  N = TransformNotmalToWorld(vec3(0.0, 0.0, 1.0));
    const float Dist = length(In.wSpacePosition.xyz);
    vec2 o = N.xy / (abs(N.x) + abs(N.y) + abs(N.z));
    if (N.z < 0.0) o = (1.0 - abs(o.yx)) * vec2(o.x >= 0.0 ? 1.0 : -1.0, o.y >= 0.0 ? 1.0 : -1.0);
    const uvec2 Oq = uvec2(clamp(o * 0.5 + 0.5, 0.0, 1.0) * 63.0 + 0.5);
    const uint  Guide = (Oq.x << 16) | (Oq.y << 22);

    // Kind: 0 fresh by the pattern, 1 reprojected, 2 rejected (fresh), 3 too old (fresh)
    uint  Kind = 0u;
    uvec4 Out  = uvec4(0u);
    const uint Rate = 1u << ((Hd.w >> 1) & 3u);
    const uint Tile = uint(((P.x >> 3) & 1) | (((P.y >> 3) & 1) << 1));
    if ((Hd.w & 1u) != 0u && (Tile % Rate) != (Hd.z % Rate))
    {
        Kind = 2u;
        const mat4  PrevW2C = mat4(uintBitsToFloat(GIScreen[1]), uintBitsToFloat(GIScreen[2]), uintBitsToFloat(GIScreen[3]), uintBitsToFloat(GIScreen[4]));
        const vec3  Prev    = In.wSpacePosition.xyz + uintBitsToFloat(GIScreen[5].xyz);
        const vec4  C       = PrevW2C * vec4(Prev, 1.0);
        if (C.w > 1e-6)
        {
            const ivec2 Q = ivec2(floor((C.xy / C.w * 0.5 + 0.5) * vec2(float(W), float(H))));
            if (Q.x >= 0 && Q.y >= 0 && uint(Q.x) < W && uint(Q.y) < H)
            {
                const uvec4 S   = GIScreenPrev[screen_header_v + uint(Q.y) * W + uint(Q.x)];
                const float Sd  = unpackHalf2x16(S.y).y, Pd = length(Prev);
                const uint  Age = (S.w >> 28) & 3u;
                if ((S.w & (1u << 30)) != 0u && abs(Sd - Pd) < 0.02 * Pd + 0.01 && dot(ScreenNormal(S.w), N) > 0.9)
                {
                    if (Age >= screen_max_age_v) Kind = 3u;
                    else
                    {
                        // (its distance: this frame's, the upsample compares it; its normal: this pixel's; one frame older)
                        Kind = 1u;
                        Out  = uvec4(S.x, packHalf2x16(vec2(unpackHalf2x16(S.y).x, Dist)), S.z, (S.w & 0xC000FFFFu) | Guide | ((Age + 1u) << 28));
                    }
                }
            }
        }
    }
    if (Kind != 1u)
    {
        const vec3 V = normalize(UBOLight.wSpaceEyePos.xyz - In.wSpacePosition.xyz);
        GIPixelGI(vec3(0.0, 0.0, 1.0), N, V, vec3(1.0), 0.0, vec3(0.0));
        const bool  bE = g_GI.w > 0.5, bR = g_GIRefl.w > 0.5;
        const vec3  E = bE ? g_GI.rgb : vec3(0.0), R = bR ? g_GIRefl.rgb : vec3(0.0);
        Out = uvec4(packHalf2x16(E.rg), packHalf2x16(vec2(E.b, Dist)), packHalf2x16(R.rg),
                    (packHalf2x16(vec2(R.b, 0.0)) & 0xFFFFu) | Guide | (bE ? 1u << 30 : 0u) | (bR ? 1u << 31 : 0u));
    }
    if (bIn) GIScreen[screen_header_v + uint(P.y) * W + uint(P.x)] = Out;

    // The counters (instance::getGIStats): a subgroup's sums, one atomic each
    const uvec4 Mine = bIn ? uvec4(Kind == 0u ? 1u : 0u, Kind == 1u ? 1u : 0u, Kind == 2u ? 1u : 0u, Kind == 3u ? 1u : 0u) : uvec4(0u);
    const uvec4 Sum  = subgroupAdd(Mine);
    if (subgroupElect())
    {
        if (Sum.x != 0u) atomicAdd(GIScreen[6].x, Sum.x);
        if (Sum.y != 0u) atomicAdd(GIScreen[6].y, Sum.y);
        if (Sum.z != 0u) atomicAdd(GIScreen[6].z, Sum.z);
        if (Sum.w != 0u) atomicAdd(GIScreen[6].w, Sum.w);
    }
}
