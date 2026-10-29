#version 450
// The GI gather (gi_gbuffer_gather_plan.md, stage 2): the half-resolution GI read without geometry - a full-screen pass a
// GI buffer, at half size, on the pixels the classification gave it (xlionrender_gi_gather_vert.glsl: tested EQUAL). The surface from
// the G-buffer prepass (xlionrender_gi_gather_common.glsl): its position from the depth, into mesh space through its instance's
// transform; its normal; its GI reference from the table. The read is mb_standard_pbr.frag GIPixelGI's per-pixel one (the surfels, then
// the local mean; no vertex read: no vertices here), into the same buffer the scene pass upsamples, with the same temporal reuse
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#include "xlionrender_surfel_gi.glsl"
#include "xlionrender_gi_gather_common.glsl"

layout(early_fragment_tests) in;
struct InstanceData { vec4 Rows[3]; vec4 Material[3]; vec4 PrevRows[3]; };     // (xlionrender_static_geom_body.glsl)
layout(std430, set = 1, binding = 1) readonly buffer InstanceBuffer { InstanceData Instances[]; } instances;
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
    const ivec2 P = ivec2(gl_FragCoord.xy);
    const gather_sample S = GatherSample(P);

    // Mesh space: world = A local + t + the instance buffer's origin (the rows of A | t)
    const InstanceData I  = instances.Instances[S.bHit ? S.Slot : 0u];
    const mat3  At = mat3(I.Rows[0].xyz, I.Rows[1].xyz, I.Rows[2].xyz);
    const mat3  Ai = inverse(transpose(At));
    const vec3  Off = vec3(uintBitsToFloat(GITable[push.TableBase + 16u]), uintBitsToFloat(GITable[push.TableBase + 17u]), uintBitsToFloat(GITable[push.TableBase + 18u]));
    const vec3  Pm = Ai * (S.P - Off - vec3(I.Rows[0].w, I.Rows[1].w, I.Rows[2].w));
    const vec3  GFaceD = cross(dFdx(Pm), dFdy(Pm));          // (before any lane leaves: the read's face normal)
    if (!S.bHit) return;

    const uvec4 Hd = GIScreen[0];
    const uint  W = Hd.x, H = Hd.y;
    const bool  bIn = uint(P.x) < W && uint(P.y) < H;
    const vec3  N = S.N;
    const float Dist = length(S.P);
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
        const vec3  Prev    = S.P + uintBitsToFloat(GIScreen[5].xyz);
        const vec4  C       = PrevW2C * vec4(Prev, 1.0);
        if (C.w > 1e-6)
        {
            const ivec2 Q = ivec2(floor((C.xy / C.w * 0.5 + 0.5) * vec2(float(W), float(H))));
            if (Q.x >= 0 && Q.y >= 0 && uint(Q.x) < W && uint(Q.y) < H)
            {
                const uvec4 R   = GIScreenPrev[screen_header_v + uint(Q.y) * W + uint(Q.x)];
                const float Sd  = unpackHalf2x16(R.y).y, Pd = length(Prev);
                const uint  Age = (R.w >> 28) & 3u;
                if ((R.w & (1u << 30)) != 0u && abs(Sd - Pd) < 0.02 * Pd + 0.01 && dot(ScreenNormal(R.w), N) > 0.9)
                {
                    if (Age >= screen_max_age_v) Kind = 3u;
                    else
                    {
                        Kind = 1u;
                        Out  = uvec4(R.x, packHalf2x16(vec2(unpackHalf2x16(R.y).x, Dist)), R.z, (R.w & 0xC000FFFFu) | Guide | ((Age + 1u) << 28));
                    }
                }
            }
        }
    }
    if (Kind != 1u)
    {
        // mb_standard_pbr.frag GIPixelGI, the per-pixel read (inNormal (0, 0, 1): the surface's own normal). ponytail: a copy of its
        // policy until the material compiler's step (stage 2c) moves it into xlionrender_surfel_gi.glsl for both
        const uint Offset = S.Entry.x, Ref = S.Entry.y;
        const vec3 GN0    = normalize(At * N);
        vec3       GFace  = dot(GFaceD, GFaceD) > 1e-24 ? normalize(dot(GFaceD, GN0) < 0.0 ? -GFaceD : GFaceD) : GN0;
        if (dot(GFace, GN0) < 0.7) GFace = GN0;
        vec4 E = vec4(0.0);
        g_GIRefl = vec4(0.0);
        if ((Ref & gi_per_pixel_v) != 0u)
        {
            g_GIMoments      = (Ref & gi_no_moments_v) == 0u;
            g_GIPixelCap     = GIPixelCap(Ref);
            g_GINoCorners    = (Ref & 0x01000000u) != 0u;
            g_GIAblateLookup = (Ref >> 25u) & 3u;
            g_GIReflDir      = normalize(Ai * reflect(normalize(S.P), N));        // (the eye at the origin: V = -P)
            const GIQueryResult Q = SurfelGIQuery(Pm, GFace, GN0, Offset, Ref & 0xFFu);
            if (Q.bSupported) E = vec4(Q.Irradiance, 1.0);
            if (!Q.bSupported || Q.Confidence < 0.25) g_GIRefl = vec4(0.0);
        }
        if (E.w < 0.5 && (Ref & 0x8000u) == 0u) E = vec4(SurfelGILocalMean(Pm, GFace, Offset, Ref & 0xFFu), 1.0);
        const bool bE = E.w > 0.5, bR = g_GIRefl.w > 0.5;
        const vec3 Rr = bR ? g_GIRefl.rgb : vec3(0.0);
        Out = uvec4(packHalf2x16(E.rg), packHalf2x16(vec2(E.b, Dist)), packHalf2x16(Rr.rg),
                    (packHalf2x16(vec2(Rr.b, 0.0)) & 0xFFFFu) | Guide | (bE ? 1u << 30 : 0u) | (bR ? 1u << 31 : 0u));
    }
    if (bIn) GIScreen[screen_header_v + uint(P.y) * W + uint(P.x)] = Out;

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
