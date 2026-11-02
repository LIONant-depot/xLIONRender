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
layout(std430, set = 1, binding = 20) readonly buffer AOBuffer { vec2 AOData[]; };    // the ambient occlusion on this grid (xlionrender_ao_comp.glsl): AO, distance

// The AO at this pixel, its 3 x 3 neighbours weighed by their distance's match (the denoiser's spatial pass: edge-aware)
float GatherAO(in const ivec2 P, in const uint W, in const uint H, in const float Dist)
{
    float S = 0.0, Wt = 0.0;
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x)
    {
        const ivec2 q = clamp(P + ivec2(x, y), ivec2(0), ivec2(int(W) - 1, int(H) - 1));
        const vec2  A = AOData[uint(q.y) * W + uint(q.x)];
        if (A.y <= 0.0) continue;
        const float w = max(0.0, 1.0 - abs(A.y - Dist) / (0.02 * Dist + 0.01)) * ((x == 0 && y == 0) ? 2.0 : 1.0);
        S += A.x * w; Wt += w;
    }
    return Wt > 1e-4 ? S / Wt : 1.0;
}

// Experiment X1 (XLION_GI_CLIP, xlionrender_gi_clip_comp.glsl): the light at a world point for a normal, from the clipmap: the finest level
// holding it, 8 cells x the 3 faces the normal looks toward (squared cosines), the weighted light over the weights - cells no surfel
// reaches count for nothing. A level's edge blends into the next
layout(std430, set = 1, binding = 21) readonly buffer GIClipBuffer { uvec4 GIClip[]; };
layout(std430, set = 1, binding = 22) buffer GIRefSampleBuffer { vec4 GIRefSample[]; };        // the ground truth (XLION_GI_REF, xlionrender_gi_ref_comp.glsl)
layout(std430, set = 1, binding = 23) readonly buffer GIRefAccumBuffer { vec4 GIRefAccum[]; };
const int clip_dim_v = 64, clip_levels_v = 4;
float TF(in const uint i) { return uintBitsToFloat(GITable[push.TableBase + i]); }
bool g_ClipFloat = false;      // (experiment X1c: the faces hold floats)
vec4 ClipLevel(in const int L, in const vec3 Pw, in const vec3 n, in const vec3 EyeL, in const float Cell0)
{
    const float Cell = Cell0 * float(1 << L);
    const ivec3 Org  = ivec3(floor(EyeL / Cell)) - ivec3(clip_dim_v / 2);
    const vec3  g    = Pw / Cell - vec3(Org) - 0.5;
    if (any(lessThan(g, vec3(0.0))) || any(greaterThanEqual(g, vec3(clip_dim_v - 1)))) return vec4(0.0);
    const ivec3 i = ivec3(floor(g));
    const vec3  f = g - vec3(i);
    vec3  Sum = vec3(0.0);
    float SumW = 0.0;
    for (int a = 0; a < 3; ++a)
    {
        const float wa = n[a] * n[a];
        if (wa < 0.02) continue;
        const uint Face = uint(2 * a + (n[a] < 0.0 ? 1 : 0));
        vec3  Num = vec3(0.0);
        float Den = 0.0;
        for (int c = 0; c < 8; ++c)
        {
            const ivec3 o = ivec3(c & 1, (c >> 1) & 1, (c >> 2) & 1), q = i + o;
            const float t = (o.x != 0 ? f.x : 1.0 - f.x) * (o.y != 0 ? f.y : 1.0 - f.y) * (o.z != 0 ? f.z : 1.0 - f.z);
            const uint  e = (uint(L) * uint(clip_dim_v * clip_dim_v * clip_dim_v) + uint(q.x) + uint(q.y) * uint(clip_dim_v) + uint(q.z) * uint(clip_dim_v * clip_dim_v)) * 6u + Face;
            const uvec4 v = GIClip[e];
            if (g_ClipFloat) { Num += uintBitsToFloat(v.xyz) * t; Den += uintBitsToFloat(v.w) * t; }
            else             { Num += vec3(v.xyz) * t; Den += float(v.w) * t; }
        }
        if (Den > (g_ClipFloat ? 1e-3 : 1.0)) { Sum += Num / Den * wa; SumW += wa; }
    }
    return SumW > 0.0 ? vec4(Sum / SumW, SumW) : vec4(0.0);
}
vec4 ClipRead(in const vec3 Pcam, in const vec3 n)
{
    const vec3 EyeL = vec3(TF(22u), TF(23u), TF(24u)), Eye = vec3(TF(25u), TF(26u), TF(27u));
    const vec3 Pw   = Pcam + Eye;
    for (int L = 0; L < clip_levels_v; ++L)
    {
        const vec4 R = ClipLevel(L, Pw, n, EyeL, TF(21u));
        if (R.w > 0.5) return vec4(R.xyz, 1.0);
    }
    return vec4(0.0);
}

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
    // The read's face normal from the derivatives of a position at the same texel of every block (the checkerboard's alternate texels
    // skewed them: a wrong face on curved surfaces, sunlit surfels behind read - Tomas's Sponza arch), before any lane leaves
    vec3 Pd = Pm;
    {
        const ivec2 Size = textureSize(GPreDepth, 0);
        const ivec2 Qd   = min(P * int(push.Div) + ivec2(int(push.Div) / 2), Size - 1);
        const float zd   = texelFetch(GPreDepth, Qd, 0).r;
        const mat4  C2W  = mat4(uintBitsToFloat(uvec4(GITable[push.TableBase + 0u], GITable[push.TableBase + 1u], GITable[push.TableBase + 2u], GITable[push.TableBase + 3u])),
                                uintBitsToFloat(uvec4(GITable[push.TableBase + 4u], GITable[push.TableBase + 5u], GITable[push.TableBase + 6u], GITable[push.TableBase + 7u])),
                                uintBitsToFloat(uvec4(GITable[push.TableBase + 8u], GITable[push.TableBase + 9u], GITable[push.TableBase + 10u], GITable[push.TableBase + 11u])),
                                uintBitsToFloat(uvec4(GITable[push.TableBase + 12u], GITable[push.TableBase + 13u], GITable[push.TableBase + 14u], GITable[push.TableBase + 15u])));
        const vec4  Wd   = C2W * vec4((vec2(Qd) + 0.5) / vec2(Size) * 2.0 - 1.0, zd, 1.0);
        const float dd   = texelFetch(GPreId, Qd, 0).z;
        const vec3  Pw   = dd > 0.0 ? normalize(Wd.xyz / Wd.w) * dd : Wd.xyz / Wd.w;
        if (zd < 1.0) Pd = Ai * (Pw - Off - vec3(I.Rows[0].w, I.Rows[1].w, I.Rows[2].w));
    }
    const vec3  GFaceD = cross(dFdx(Pd), dFdy(Pd));
    if (!S.bHit) return;

    const uvec4 Hd = GIScreen[0];
    const uint  W = Hd.x, H = Hd.y;
    const bool  bIn = uint(P.x) < W && uint(P.y) < H;
    const vec3  Nr = S.N;                                   // (the vertex normal: the read's)
    const vec3  N  = S.bFront ? Nr : -Nr;                   // (its visible side: the guide's, the reflection's)
    const float Dist = length(S.P);
    vec2 o = N.xy / (abs(N.x) + abs(N.y) + abs(N.z));
    if (N.z < 0.0) o = (1.0 - abs(o.yx)) * vec2(o.x >= 0.0 ? 1.0 : -1.0, o.y >= 0.0 ? 1.0 : -1.0);
    const uvec2 Oq = uvec2(clamp(o * 0.5 + 0.5, 0.0, 1.0) * 63.0 + 0.5);
    const uint  Guide = (Oq.x << 16) | (Oq.y << 22);

    // Kind: 0 fresh by the pattern, 1 reprojected, 2 rejected (fresh), 3 too old (fresh)
    uint  Kind = 0u;
    uvec4 Out  = uvec4(0u);
    bool  bKeep = false;
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
        const vec3 GN0    = normalize(At * Nr);
        vec3       GFace  = dot(GFaceD, GFaceD) > 1e-24 ? normalize(dot(GFaceD, GN0) < 0.0 ? -GFaceD : GFaceD) : GN0;
        if (dot(GFace, GN0) < 0.7) GFace = GN0;
        vec4 E = vec4(0.0);
        bool bWeak = true;            // (no surfel supports it: the local mean - flagged, the upsample prefers its supported neighbours)
        g_GIRefl = vec4(0.0);
        if (GITable[push.TableBase + 20u] == 2u)                   // the ground truth: this pixel's point for the tracer, its mean so far (no reuse, no reflection)
        {
            const uint Rw = GITable[push.TableBase + 28u], Ri = uint(P.y) * Rw + uint(P.x);
            const vec3 Pwr = S.P + vec3(TF(25u), TF(26u), TF(27u));
            GIRefSample[2u * Ri] = vec4(Pwr, 1.0);
            GIRefSample[2u * Ri + 1u] = vec4(N, 0.0);
            const vec4 Ac = GIRefAccum[2u * Ri], Pa = GIRefAccum[2u * Ri + 1u];
            if (Ac.w > 0.5 && Pa.w > 0.5 && dot(Pa.xyz - Pwr, Pa.xyz - Pwr) < 1e-4) { E = vec4(Ac.rgb / Ac.w, 1.0); bWeak = false; }
            else E = vec4(0.0, 0.0, 0.0, 1.0);
        }
        else if (GITable[push.TableBase + 20u] == 3u)                   // experiment X1c: voxels of light on the surfaces, this pixel a place rays start from
        {
            const uint Rw = GITable[push.TableBase + 28u], Ri = uint(P.y) * Rw + uint(P.x);
            const vec3 Eyw = vec3(TF(25u), TF(26u), TF(27u));
            GIRefSample[2u * Ri] = vec4(S.P + Eyw, 1.0);
            GIRefSample[2u * Ri + 1u] = vec4(N, 0.0);
            GIRefSample[2u * W * H] = vec4(TF(22u), TF(23u), TF(24u), TF(21u));
            g_ClipFloat = true;
            E = ClipRead(S.P, N);
            bWeak = E.w < 0.5;
            E.w = 1.0;
        }
        else if (GITable[push.TableBase + 20u] != 0u)                   // experiment X1: the clipmap, both reads
        {
            E = ClipRead(S.P, N);
            bWeak = E.w < 0.5;
            const vec4 Rc = ClipRead(S.P, reflect(normalize(S.P), N));
            g_GIRefl = Rc;
            if (E.w < 0.5) E = vec4(SurfelGILocalMean(Pm, GFace, Offset, Ref & 0xFFu), 1.0);
        }
        else if (Ref == gi_probes_v) E = SurfelGIProbes(S.P, N);        // (a skinned surface: the probe grid, GIMesh its buffer)
        else if ((Ref & gi_per_pixel_v) != 0u)
        {
            g_GIMoments      = (Ref & gi_no_moments_v) == 0u;
            g_GIPixelCap     = GIPixelCap(Ref);
            g_GINoCorners    = (Ref & 0x01000000u) != 0u;
            g_GIAblateLookup = (Ref >> 25u) & 3u;
            g_GIReflDir      = normalize(Ai * reflect(normalize(S.P), N));        // (the eye at the origin: V = -P)
            const GIQueryResult Q = SurfelGIQuery(Pm, GFace, GN0, Offset, Ref & 0xFFu);
            // (the full read's policy: a low confidence - the surfels near mostly out of sight - blends into a smooth read; there the vertex
            // read, here the local mean: unblended, a weak read's noise and leaks showed as speckled patches - Tomas's Sponza arch)
            bWeak = !Q.bSupported;
            if (Q.bSupported) E = vec4(mix(SurfelGILocalMean(Pm, GFace, Offset, Ref & 0xFFu), Q.Irradiance, smoothstep(0.0, 0.25, Q.Confidence)), 1.0);
            if (!Q.bSupported || Q.Confidence < 0.25) g_GIRefl = vec4(0.0);
        }
        if (E.w < 0.5 && (Ref & 0x8000u) == 0u) E = vec4(SurfelGILocalMean(Pm, GFace, Offset, Ref & 0xFFu), 1.0);
        if (push.AO != 0u) E.rgb *= GatherAO(P, W, H, Dist);
        if (any(isnan(E.rgb)) || any(isinf(E.rgb))) E = vec4(0.0);     // (never a non-finite entry: the reprojection would carry it on)      // (a fresh read only: a reprojected one carries last frame's)
        if (GITable[push.TableBase + 29u] != 0u) g_GIRefl = vec4(0.0);
        const bool bE = E.w > 0.5, bR = g_GIRefl.w > 0.5;
        bKeep = !bE && !bR;           // (no GI here: the AO's own entry stays)
        const vec3 Rr = bR ? g_GIRefl.rgb : vec3(0.0);
        Out = uvec4(packHalf2x16(E.rg), packHalf2x16(vec2(E.b, Dist)), packHalf2x16(Rr.rg),
                    (packHalf2x16(vec2(Rr.b, 0.0)) & 0xFFFFu) | Guide | (bE ? 1u << 30 : 0u) | (bR ? 1u << 31 : 0u) | (bE && bWeak ? 3u << 28 : 0u));     // (age 3, never a fresh one's: weak - a hole in the surfels' coverage, the Cornell cube's dashes)
    }
    if (bIn && !(bKeep && push.AO != 0u)) GIScreen[screen_header_v + uint(P.y) * W + uint(P.x)] = Out;

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
