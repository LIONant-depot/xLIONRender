#version 450
// The GI gather (gi_gbuffer_gather_plan.md, stage 2): the half-resolution GI read without geometry - a full-screen pass a
// GI buffer, at half size, on the pixels the classification gave it (xlionrender_gi_gather_vert.glsl: tested EQUAL). The surface from
// the G-buffer prepass (xlionrender_gi_gather_common.glsl): its position from the depth, into mesh space through its instance's
// transform; its normal; its GI reference from the table. The read is mb_standard_pbr.frag GIPixelGI's per-pixel one (the surfels, then
// the local mean; no vertex read: no vertices here), into the same buffer the scene pass upsamples, with the same temporal reuse
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#ifdef XLION_HEAT
#extension GL_ARB_shader_clock : require
#endif
#include "xlionrender_surfel_gi.glsl"
#include "xlionrender_gi_gather_common.glsl"

layout(early_fragment_tests) in;
struct InstanceData { vec4 Rows[3]; vec4 Material[3]; vec4 PrevRows[3]; };     // (xlionrender_static_geom_body.glsl)
layout(std430, set = 1, binding = 1) readonly buffer InstanceBuffer { InstanceData Instances[]; } instances;
layout(std430, set = 1, binding = 11) buffer GIScreenBuffer { uvec4 GIScreen[]; };
layout(std430, set = 1, binding = 13) readonly buffer GIScreenPrevBuffer { uvec4 GIScreenPrev[]; };      // (last frame's history: what stands still only)
layout(std430, set = 1, binding = 26) writeonly buffer GIScreenHistoryBuffer { uvec4 GIScreenHistory[]; };   // (this frame's: Tomas 2026-11-13 - a mover's pixel keeps the static entry under it)
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
// (XLION_SSGI, push.AO 2: the AO pass's screen GI at this pixel, the near field's irradiance - after the AO's 4 x 4 filter)
vec3 GatherSSGI(in const ivec2 P, in const uint W, in const uint H)
{
    const uint N = W * H, i = uint(P.y) * W + uint(P.x);
    return max(vec3(AOData[N + i], AOData[2u * N + i].x), vec3(0.0));
}

// Experiment X1 (XLION_GI_CLIP, xlionrender_gi_clip_comp.glsl): the light at a world point for a normal, from the clipmap: the finest level
// holding it, 8 cells x the 3 faces the normal looks toward (squared cosines), the weighted light over the weights - cells no surfel
// reaches count for nothing. A level's edge blends into the next
layout(std430, set = 1, binding = 21) readonly buffer GIClipBuffer { uvec4 GIClip[]; };
layout(std430, set = 1, binding = 22) buffer GIRefSampleBuffer { vec4 GIRefSample[]; };        // the ground truth (XLION_GI_REF, xlionrender_gi_ref_comp.glsl)
layout(std430, set = 1, binding = 23) readonly buffer GIRefAccumBuffer { vec4 GIRefAccum[]; };
// (XLION_VOX_PUBLISHED: binding 24 the published light - xlionrender_vox_levels.glsl)
layout(std430, set = 1, binding = 24) buffer PubBuffer { uvec4 Pub[]; };
layout(std430, set = 1, binding = 24) buffer PubBufferU { uint PubU[]; };
#ifdef XLION_HEAT
layout(std430, set = 1, binding = 25) buffer HeatBuffer { uint Heat[]; };     // XLION_HEAT: this pass's clock a pixel (xlionrender_heat_comp.glsl), after a 4-uint header
#endif
#define VOX_READ_SCREEN
#define VOX_PUBLISHED_BUFFER
#include "xlionrender_vox_levels.glsl"
#include "xlionrender_sh_l1.glsl"
// Milestone A (milestone_a_impl_2026-11-13.md): a skinned pixel's light from its mover's probes - 2 x 3 x 2, trilinear, each by its validity;
// the lattice from the gather table (GITable[30]: where, 8 a block), the probes from the store's tail (xlionrender_gi_ref.glsl mode -8).
// The all-invalid contract (plan v3.1 #1): none valid -> the invalid ones' last estimates if no older than 0.5 s (w 0.5: historical), else w 0 and
// Centre = the support's centre (world) for the caller's coarse read of the static store
vec4 MoverProbesRead(in const vec3 Pw, in const vec3 n, in const uint b, out vec3 Centre)
{
    const uint Lt = push.TableBase + GITable[push.TableBase + 30u] + 8u * b;
    const vec3 O = vec3(uintBitsToFloat(GITable[Lt]), uintBitsToFloat(GITable[Lt + 1u]), uintBitsToFloat(GITable[Lt + 2u]));
    const float Cxz = uintBitsToFloat(GITable[Lt + 3u]), Cy = uintBitsToFloat(GITable[Lt + 4u]);
    Centre = O + vec3(0.5 * Cxz, Cy, 0.5 * Cxz);
    if (Cxz <= 0.0 || Cy <= 0.0) return vec4(0.0);
    const vec3 g = clamp((Pw - O) / vec3(Cxz, Cy, Cxz), vec3(0.0), vec3(1.0, 2.0, 1.0));
    const uint iy = min(uint(g.y), 1u);
    const vec3 f = vec3(g.x, g.y - float(iy), g.z);
    vec4 R = vec4(0.0), G = vec4(0.0), B = vec4(0.0), Rh = vec4(0.0), Gh = vec4(0.0), Bh = vec4(0.0);
    float W = 0.0, Wh = 0.0;
    for (uint c = 0u; c < 8u; ++c)
    {
        const uvec3 o = uvec3(c & 1u, (c >> 1u) & 1u, (c >> 2u) & 1u);
        const uint  e = vox_probe_base_v + 4u * (12u * b + o.x + 2u * (iy + o.y) + 6u * o.z);
        const vec4  V = uintBitsToFloat(GIClip[e + 3u]);
        if (V.y <= 0.0) continue;
        const float t = (o.x != 0u ? f.x : 1.0 - f.x) * (o.y != 0u ? f.y : 1.0 - f.y) * (o.z != 0u ? f.z : 1.0 - f.z), w = t * V.x;
        const vec4  Sr = uintBitsToFloat(GIClip[e]), Sg = uintBitsToFloat(GIClip[e + 1u]), Sb = uintBitsToFloat(GIClip[e + 2u]);
        if (w > 0.0) { R += Sr * w; G += Sg * w; B += Sb * w; W += w; }
        else if (V.z <= 0.5) { const float th = max(t, 1e-4); Rh += Sr * th; Gh += Sg * th; Bh += Sb * th; Wh += th; }
    }
    if (W > 1e-3) return vec4(ShIrradiance(R / W, G / W, B / W, n), 1.0);
    return Wh > 0.0 ? vec4(ShIrradiance(Rh / Wh, Gh / Wh, Bh / Wh, n), 0.5) : vec4(0.0);
}

#ifdef VOX_PUBLISHED_BUFFER
// The published read (word 37 1): this level's 8 corners a face from the published entries; any corner not resolved this frame or the last
// (a new one: its request listed for the next resolve) -> bMiss, the caller reads the levels as before. Every corner it uses asks to be
// resolved again (a bit a slot a frame: listed once)
vec4 VoxReadPublished(in const vec3 Pcam, in const vec3 n, out bool bMiss)
{
    bMiss = false;
    const vec3  EyeL = vec3(TF(22u), TF(23u), TF(24u)), Pw = Pcam + vec3(TF(25u), TF(26u), TF(27u));
    const float Cell0 = TF(21u);
    const int   L = PubLevel(Pw, EyeL, Cell0);
    if (L < 0) return vec4(0.0);
    const uint  Frame = Pub[3].x, Parity = Frame & 1u;      // (the CPU's, copied in this frame)
    const float Cell = Cell0 * float(1 << L);
    const ivec3 Org  = ivec3(floor(EyeL / Cell)) - ivec3(clip_dim_v / 2);
    const vec3  g    = Pw / Cell - vec3(Org) - 0.5;
    vec3 Sum = vec3(0.0); float SumW = 0.0;
    for (int a = 0; a < 3; ++a)
    {
        const float wa = n[a] * n[a];
        if (wa < 0.02) continue;
        const uint Face = uint(2 * a + (n[a] < 0.0 ? 1 : 0)) + 64u * g_VoxBucket;
        vec3 gv = vec3(Org) + g; gv[a] *= vox_layers_v;
        const ivec3 iv = ivec3(floor(gv));
        const vec3  fc = gv - vec3(iv);
        vec3 Num = vec3(0.0); float Den = 0.0;
        for (int c = 0; c < 8; ++c)
        {
            const ivec3 o = ivec3(c & 1, (c >> 1) & 1, (c >> 2) & 1), q = iv + o;
            const float t = (o.x != 0 ? fc.x : 1.0 - fc.x) * (o.y != 0 ? fc.y : 1.0 - fc.y) * (o.z != 0 ? fc.z : 1.0 - fc.z);
            const uint  h = VoxHash(L, q, Face), s = h % vox_slots_v;
            // (the entry first: a corner resolved this frame or the last asks nothing - the requests' atomics only for older or missing ones,
            // resolved again every two frames while read; read up to 3 frames old)
            const uvec4 P = Pub[pub_entry0_v + s];
            const uint  Age = Frame - (P.w & 0x7FFFFFFFu);
            const bool  bOurs = uintBitsToFloat(P.z) == VoxTagOf(h);
            if (!bOurs || Age >= 2u)
            {
                const uint Bit = 1u << (s & 31u);
                if ((atomicOr(PubU[pub_head_v * 4u + (s >> 5u)], Bit) & Bit) == 0u)
                {
                    const uint k = atomicAdd(PubU[Parity * 4u], 1u);
                    if (k < pub_cap_v) Pub[pub_list0_v + Parity * pub_cap_v + k] = uvec4(uvec3(q), (uint(L) << 8u) | Face);
                    else atomicAdd(PubU[2u * 4u + 2u], 1u);
                }
            }
            if (!bOurs || Age > 3u) { bMiss = true; continue; }
            if ((P.w & 0x80000000u) == 0u) continue;          // (resolved: nothing there)
            Num += vec3(unpackHalf2x16(P.x), unpackHalf2x16(P.y).x) * t; Den += t;
        }
        if (Den > 1e-3) { Sum += Num / Den * wa; SumW += wa; }
    }
    return SumW > 0.0 ? vec4(Sum / SumW, 1.0) : vec4(0.0);
}
#endif
#ifdef XLION_GI_BRICKS
// B1 (XLION_GI_BRICKS, xlionrender_gi_brick.glsl): the store's first two levels read from their published 3D textures (bindings 13-28: level
// 0's 8 volumes, then level 1's), 3 filtered samples a level instead of 8 hashed corners a face; the coarser levels from the store
layout(binding = 13) uniform sampler3D GIBrick0;
layout(binding = 14) uniform sampler3D GIBrick1;
layout(binding = 15) uniform sampler3D GIBrick2;
layout(binding = 16) uniform sampler3D GIBrick3;
layout(binding = 17) uniform sampler3D GIBrick4;
layout(binding = 18) uniform sampler3D GIBrick5;
layout(binding = 19) uniform sampler3D GIBrick6;
layout(binding = 20) uniform sampler3D GIBrick7;
layout(binding = 21) uniform sampler3D GIBrick8;
layout(binding = 22) uniform sampler3D GIBrick9;
layout(binding = 23) uniform sampler3D GIBrick10;
layout(binding = 24) uniform sampler3D GIBrick11;
layout(binding = 25) uniform sampler3D GIBrick12;
layout(binding = 26) uniform sampler3D GIBrick13;
layout(binding = 27) uniform sampler3D GIBrick14;
layout(binding = 28) uniform sampler3D GIBrick15;
vec4 BrickTex(in const int L, in const uint i, in const vec3 u)
{
    switch (uint(L) * 8u + i)
    {
    case 0u: return textureLod(GIBrick0, u, 0.0);
    case 1u: return textureLod(GIBrick1, u, 0.0);
    case 2u: return textureLod(GIBrick2, u, 0.0);
    case 3u: return textureLod(GIBrick3, u, 0.0);
    case 4u: return textureLod(GIBrick4, u, 0.0);
    case 5u: return textureLod(GIBrick5, u, 0.0);
    case 6u: return textureLod(GIBrick6, u, 0.0);
    case 7u: return textureLod(GIBrick7, u, 0.0);
    case 8u: return textureLod(GIBrick8, u, 0.0);
    case 9u: return textureLod(GIBrick9, u, 0.0);
    case 10u: return textureLod(GIBrick10, u, 0.0);
    case 11u: return textureLod(GIBrick11, u, 0.0);
    case 12u: return textureLod(GIBrick12, u, 0.0);
    case 13u: return textureLod(GIBrick13, u, 0.0);
    case 14u: return textureLod(GIBrick14, u, 0.0);
    default: return textureLod(GIBrick15, u, 0.0);
    }
}
#include "xlionrender_gi_brick.glsl"
const int gi_brick_levels_v = 2;        // (instance::gi_brick_levels_v)
// VoxReadLevels with the bricks' levels: the same window test, the same blend by trust - a brick level's trust its filtered validity
// (ponytail: trusted at 32 samples, the store's read at 256 - young records weigh more than there; a count volume if the grade asks)
vec4 VoxReadBricks(in const vec3 Pcam, in const vec3 n)
{
    const vec3 EyeL = vec3(TF(22u), TF(23u), TF(24u)), Pw = Pcam + vec3(TF(25u), TF(26u), TF(27u));
    vec3 Sum = vec3(0.0), Last = vec3(0.0); float Left = 1.0; bool bAny = false;
    for (int L = 0; L < clip_levels_v && Left > 1e-3; ++L)
    {
        vec4 R;
        if (L < gi_brick_levels_v)
        {
            const float Cell = TF(21u) * float(1 << L);
            const vec3  g    = Pw / Cell - vec3(ivec3(floor(EyeL / Cell)) - ivec3(clip_dim_v / 2)) - 0.5;
            if (any(lessThan(g, vec3(0.0))) || any(greaterThanEqual(g, vec3(clip_dim_v - 1)))) continue;
            R = BrickRead(L, Pw, n);
            const vec3 Inside = min(g, vec3(clip_dim_v - 1) - g);
            g_ClipTrust = clamp(g_BrickV, 0.0, 1.0) * clamp(min(Inside.x, min(Inside.y, Inside.z)) / 4.0, 0.0, 1.0);
        }
        else R = ClipLevel(L, Pw, n, EyeL, TF(21u));
        if (R.w <= 0.0) continue;
        Sum += R.xyz * (Left * g_ClipTrust); Left *= 1.0 - g_ClipTrust;
        Last = R.xyz; bAny = true;
    }
    return bAny ? vec4(Sum + Last * Left, 1.0) : vec4(0.0);
}
#endif
// The light around the mirror direction, from the rays this place's own face sent that way (xlionrender_gi_ref_comp.glsl VoxStoreDir): the three axes R leans on, by their squares, as a face read.
// Finest level that knows, the rest from its parent; what none vouches for: the place's irradiance (Fallback; EnvLighting then gives about its older ratio), so a new cell fades in
vec4 VoxBin(in const uvec4 q) { return vec4(vec3(q.xyz) / 256.0, float(q.w) / 16.0); }     // (a direction bin's fixed point sum: xlionrender_gi_ref.glsl VoxStoreDir)
vec4 VoxReflect(in const vec3 Pcam, in const vec3 n, in const vec3 Fallback)
{
    const vec3 EyeL = vec3(TF(22u), TF(23u), TF(24u)), Pw = Pcam + vec3(TF(25u), TF(26u), TF(27u));
    const vec3 Rd = reflect(normalize(Pcam), n), An = abs(n);
    const int  a = An.x >= An.y && An.x >= An.z ? 0 : An.y >= An.z ? 1 : 2, u = (a + 1) % 3, v = (a + 2) % 3;
    const uint Base = 6u + uint(2 * a + (n[a] < 0.0 ? 1 : 0)) * 5u;
    const uint  Bk[3] = uint[3](4u, Rd[u] > 0.0 ? 0u : 1u, Rd[v] > 0.0 ? 2u : 3u);
    const float Bw[3] = float[3](Rd[a] * Rd[a], Rd[u] * Rd[u], Rd[v] * Rd[v]);
    const float Cell0 = TF(21u);
    // A bin is wide: the rays of a floor's bin toward the street go up to about 55 degrees, into the sky; a reflection near the surface stays low, among the walls
    const float SkyTake = GITable[push.TableBase + 31u] != 0u ? 1.0 : smoothstep(0.0, 0.5, abs(Rd[a]));     // (no sky left out unless XLION_VOX_NO_BIN_PDF: the bin's own mean is meant to be right near the surface)
    vec3 Sum = vec3(0.0); float Left = 1.0;
    for (int L = 0; L < clip_levels_v && Left > 1e-3; ++L)
    {
        const float Cell = Cell0 * float(1 << L);
        const ivec3 Org  = ivec3(floor(EyeL / Cell)) - ivec3(clip_dim_v / 2);
        const vec3  g    = Pw / Cell - vec3(Org) - 0.5;
        if (any(lessThan(g, vec3(0.0))) || any(greaterThanEqual(g, vec3(clip_dim_v - 1)))) continue;
        const vec3  gv = vec3(Org) + g;
        const ivec3 iv = ivec3(floor(gv));
        const vec3  f  = gv - vec3(iv);
        vec3 Lit = vec3(0.0); float WSum = 0.0, Trust = 0.0;
        for (int b = 0; b < 3; ++b)
        {
            if (Bw[b] < 0.02) continue;
            vec3 Num = vec3(0.0); float Den = 0.0;
            for (int c = 0; c < 4; ++c)        // (the face's plane: along its axis the corner its samples rounded to)
            {
                ivec3 o = ivec3(0); o[u] = c & 1; o[v] = c >> 1; o[a] = f[a] >= 0.5 ? 1 : 0;
                const float t = (o[u] != 0 ? f[u] : 1.0 - f[u]) * (o[v] != 0 ? f[v] : 1.0 - f[v]);
                const uint  h = VoxHash(L, iv + o, Base + Bk[b]), e = VoxEntryOf(h);
                if (uintBitsToFloat(GIClip[e + 1u].x) == VoxTagOf(h)) { const vec4 w = VoxBin(GIClip[e]); if (w.w > 0.0) { const vec4 m0 = SkyTake < 1.0 ? VoxBin(uvec4(GIClip[VoxX2Of(e)].xyz, GIClip[e + 1u].z)) : vec4(0.0);      /* (the sky kept - the default: its bins not read, a load a corner saved) */ const vec4 m = m0 * ((1.0 - SkyTake) * smoothstep(8.0, 48.0, w.w - m0.w));      /* (the sky left out only where enough rays hit something: a few hits' mean flips between tiles: black squares, Tomas) */ const float cw = min(w.w, 128.0) * t, nl = w.w - m.w; Num += (nl > 0.5 ? (w.rgb - m.rgb) / nl : w.rgb / w.w) * cw; Den += cw; } }
            }
            if (Den > 1e-3) { Lit += Num / Den * Bw[b]; WSum += Bw[b]; Trust += Den * Bw[b]; }
        }
        if (WSum < 0.5) continue;
        const float Tr = clamp(Trust / WSum / 64.0, 0.0, 1.0);
        Sum += Lit / WSum * (Left * Tr); Left *= 1.0 - Tr;
    }
    return vec4(Sum + Fallback * Left, 1.0);
}
// XLION_VOX_DIRS (word 32: bit 3 on; bits 0, 1 level; bit 2 k 16; bits 8 to 15 roughness x 255; word 33 the frame; xlionrender_gi_ref.glsl VoxDirsAdd): the read, samples of the lobe, a position in the surface's plane by chance (bilinear over time)
uint DirsHash(inout uint s) { s ^= s >> 16; s *= 0x7FEB352Du; s ^= s >> 15; s *= 0x846CA68Bu; s ^= s >> 16; return s; }
vec4 VoxDirsRead(in const vec3 Pcam, in const vec3 n, in const vec4 Fallback, in const ivec2 Px)
{
    const uint  Cfg = GITable[push.TableBase + 32u];
    const int   L = int(Cfg & 3u); const uint k = (Cfg & 4u) != 0u ? 16u : 8u;
    const float Rg = float((Cfg >> 8) & 255u) / 255.0, al = max(Rg * Rg, 0.002), Cell = TF(21u) * float(1 << L);
    const vec3  Pw = Pcam + vec3(TF(25u), TF(26u), TF(27u)), Ve0 = -normalize(Pcam), An = abs(n);
    const int   a = An.x >= An.y && An.x >= An.z ? 0 : An.y >= An.z ? 1 : 2, u = (a + 1) % 3, v = (a + 2) % 3;
    const float sg = n[a] < 0.0 ? -1.0 : 1.0; const uint Face = uint(2 * a + (n[a] < 0.0 ? 1 : 0));
    const vec3  T = normalize(cross(abs(n.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0), n)), B = cross(n, T), Ve = vec3(dot(Ve0, T), dot(Ve0, B), max(dot(Ve0, n), 1e-3));
    const vec3  Vh = normalize(vec3(al * Ve.x, al * Ve.y, Ve.z));
    const float Lq = Vh.x * Vh.x + Vh.y * Vh.y;
    const vec3  T1 = Lq > 1e-9 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(Lq) : vec3(1.0, 0.0, 0.0), T2 = cross(Vh, T1);
    uint Seed = uint(Px.x) * 73856093u ^ uint(Px.y) * 19349663u ^ GITable[push.TableBase + 33u] * 83492791u;
    vec3 Sum = vec3(0.0);
    const bool bFixed = (Cfg & 16u) != 0u;      // (the affordable read: 4 fixed directions of the lobe, each over the 4 records around, 16 fetches, no noise)
    const int nS = bFixed ? 16 : 8;
    const vec2 Fx = fract(Pw[u] / Cell - 0.5) * vec2(1.0, 0.0) + fract(Pw[v] / Cell - 0.5) * vec2(0.0, 1.0);
    float SumW = 0.0;
    for (int i = 0; i < nS; ++i)
    {
        const float r1 = bFixed ? (float(i >> 2 & 1) + 0.5) * 0.5 : float(DirsHash(Seed)) / 4294967296.0, r2 = bFixed ? (float(i >> 3 & 1) * 0.5 + float(i >> 2 & 1) * 0.25 + 0.125) : float(DirsHash(Seed)) / 4294967296.0;
        const float j1 = bFixed ? float(i & 1) : float(DirsHash(Seed)) / 4294967296.0, j2 = bFixed ? float(i >> 1 & 1) : float(DirsHash(Seed)) / 4294967296.0;
        const float Wt = bFixed ? ((i & 1) != 0 ? Fx.x : 1.0 - Fx.x) * ((i & 2) != 0 ? Fx.y : 1.0 - Fx.y) : 1.0;
        const float r = sqrt(r1), Phi = 6.2831853 * r2, t1 = r * cos(Phi), s = 0.5 * (1.0 + Vh.z), t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * r * sin(Phi);
        const vec3  Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh, Ne = normalize(vec3(al * Nh.x, al * Nh.y, max(0.0, Nh.z)));
        vec3 Le = reflect(-Ve, Ne); if (Le.z <= 1e-4) Le.z = abs(Le.z) + 1e-4;
        const vec3  D = normalize(T * Le.x + B * Le.y + n * Le.z);
        vec3 q = Pw / Cell; if (bFixed) { q[u] = floor(q[u] - 0.5) + j1 + 0.5; q[v] = floor(q[v] - 0.5) + j2 + 0.5; } else { q[u] += j1 - 0.5; q[v] += j2 - 0.5; }
        const float sd = abs(D[u]) + abs(D[v]) + max(D[a] * sg, 0.0) + 1e-6, x = D[u] / sd, y = D[v] / sd;
        const uint  di = min(uint((x + y + 1.0) * 0.5 * float(k)), k - 1u), dj = min(uint((x - y + 1.0) * 0.5 * float(k)), k - 1u);
        const uint  h = VoxHash(L, ivec3(floor(q)), 1000u + Face * 256u + di * k + dj), e = VoxEntryOf(h);
        const bool  bOurs = uintBitsToFloat(GIClip[e + 1u].x) == VoxTagOf(h);
        const vec4  w = bOurs ? VoxBin(GIClip[e]) : vec4(0.0);
        const float Tr = clamp(w.w / 64.0, 0.0, 1.0);        // (under a few rays: what the bins say, explicitly)
        Sum += Wt * mix(Fallback.rgb, w.w > 0.0 ? w.rgb / w.w : vec3(0.0), Tr); SumW += Wt;
    }
    return vec4(Sum / max(SumW, 1e-4), 1.0);
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

const float kGIScreenBlend = 0.10;      // the fresh irradiance's share over last frame's (1: none kept); 0.25 -> 0.10 2026-11-12: panel flicker 2.9 -> 1.0 % of pixels, a sun jump ~0.5 s slower to settle
// The history read bilinearly (Tomas 2026-11-13: every GI / history buffer bilinear - the single nearest entry made the light float while the camera
// moved): the 4 entries around the point's place last frame, each kept only if it is the same surface (distance within 2 % + 1 cm, normal within
// ~25 degrees, not a mover's), weighted by the bilinear share and renormalized. E: the irradiance's (not a weak entry's), R: the reflection's;
// w 0: none. Age: the oldest used (the reuse path's limit)
void HistoryBilinear(in const vec4 C, in const uint W, in const uint H, in const float Ld, in const vec3 N, out vec4 E, out vec4 R, out uint Age)
{
    E = vec4(0.0); R = vec4(0.0); Age = 0u;
    const vec2 g = (C.xy / C.w * 0.5 + 0.5) * vec2(float(W), float(H)) - 0.5, i0 = floor(g), f = g - i0;
    for (int k = 0; k < 4; ++k)
    {
        const ivec2 Q = ivec2(i0) + ivec2(k & 1, k >> 1);
        if (Q.x < 0 || Q.y < 0 || uint(Q.x) >= W || uint(Q.y) >= H) continue;
        const float wb = ((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k >> 1) != 0 ? f.y : 1.0 - f.y);
        if (wb <= 0.0) continue;
        const uvec4 Rp = GIScreenPrev[screen_header_v + uint(Q.y) * W + uint(Q.x)];
        if (abs(unpackHalf2x16(Rp.y).y - Ld) >= 0.02 * Ld + 0.01 || dot(ScreenNormal(Rp.w), N) <= 0.9 || (Rp.w & 1u) != 0u) continue;
        const uint A = (Rp.w >> 28) & 3u;
        if ((Rp.w & (1u << 30)) != 0u && A != 3u) { E += vec4(vec3(unpackHalf2x16(Rp.x), unpackHalf2x16(Rp.y).x) * wb, wb); Age = max(Age, A); }
        if ((Rp.w & (1u << 31)) != 0u) R += vec4(vec3(unpackHalf2x16(Rp.z), unpackHalf2x16(Rp.w & 0xFFFFu).x) * wb, wb);
    }
    if (E.w > 0.0) E.rgb /= E.w;
    if (R.w > 0.0) R.rgb /= R.w;
}

void main()
{
#ifdef XLION_HEAT
    const uint HeatT0 = clock2x32ARB().x;          // (the subgroup's clock: its lanes' cost together, stalls included)
    {   // (the header's [2] subgroups, [3] their pixels: how full the passes' subgroups are - helpers left out)
        const uint nLanes = subgroupBallotBitCount(subgroupBallot(!gl_HelperInvocation));
        if (subgroupElect()) { atomicAdd(Heat[2], 1u); atomicAdd(Heat[3], nLanes); }
    }
#endif
    const ivec2 P = ivec2(gl_FragCoord.xy);
    const gather_sample S = GatherSample(P);

    // Mesh space: world = A local + t + the instance buffer's origin (the rows of A | t)
    const InstanceData I  = instances.Instances[S.bHit ? S.Slot : 0u];
    const mat3  At = mat3(I.Rows[0].xyz, I.Rows[1].xyz, I.Rows[2].xyz);
    const mat3  Ai = inverse(transpose(At));
    const vec3  Off = vec3(uintBitsToFloat(GITable[push.TableBase + 16u]), uintBitsToFloat(GITable[push.TableBase + 17u]), uintBitsToFloat(GITable[push.TableBase + 18u]));
    const vec3  Pm = Ai * (S.P - Off - vec3(I.Rows[0].w, I.Rows[1].w, I.Rows[2].w));
    // (C39: the surface's own move since last frame, by its instance's transform then (PrevRows, instance_slots - the TAA's motion): the GI history reprojects
    // a mover's point to where it was, not to where a static point would have been. 0 for what stands still. A skinned mesh's limbs move inside it: the history's mover rule)
    const vec3  Moved = Pm * (mat3(I.PrevRows[0].xyz, I.PrevRows[1].xyz, I.PrevRows[2].xyz) - At) + vec3(I.PrevRows[0].w - I.Rows[0].w, I.PrevRows[1].w - I.Rows[1].w, I.PrevRows[2].w - I.Rows[2].w);
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
    // (C39: a mover - skinned, or its transform changed since last frame - and what stands still never take each other's history: a walker's pixels took the
    // street's light behind it and the street took the walker's blotches, kept at 90 % a frame: a trail. The entry's bit 0 says which it was)
    const bool  bMover = S.Entry.y == gi_probes_v || S.Entry.y == gi_screen_v || any(notEqual(Moved, vec3(0.0)));

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
        const vec3  Prev    = S.P + uintBitsToFloat(GIScreen[5].xyz) + Moved;
        const vec4  C       = PrevW2C * vec4(Prev, 1.0);
        if (C.w > 1e-6)
        {
            vec4 He, Hr; uint Age;
            HistoryBilinear(C, W, H, length(Prev), N, He, Hr, Age);
            if (He.w > 0.5 && !bMover)      // (most of the bilinear footprint on this surface)
            {
                if (Age >= screen_max_age_v) Kind = 3u;
                else
                {
                    Kind = 1u;
                    Out  = uvec4(packHalf2x16(He.rg), packHalf2x16(vec2(He.b, Dist)), packHalf2x16(Hr.rg),
                                 (packHalf2x16(vec2(Hr.b, 0.0)) & 0xFFFEu) | Guide | (1u << 30) | (Hr.w > 0.0 ? 1u << 31 : 0u) | ((Age + 1u) << 28));
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
            GIRefSample[2u * W * H] = vec4(TF(25u), TF(26u), TF(27u), TF(30u));      // (the eye, the roughness the reflections' ground truth grades: xlionrender_gi_ref.glsl)
            const vec4 Ac = GIRefAccum[2u * Ri], Pa = GIRefAccum[2u * Ri + 1u];
            if (Ac.w > 0.5 && Pa.w > 0.5 && dot(Pa.xyz - Pwr, Pa.xyz - Pwr) < 1e-4) { E = vec4(Ac.rgb / Ac.w, 1.0); bWeak = false; }
            else E = vec4(0.0, 0.0, 0.0, 1.0);
        }
        else if (GITable[push.TableBase + 20u] == 3u)                   // experiment X1c: voxels of light on the surfaces, this pixel a place rays start from
        {
            const uint Rw = GITable[push.TableBase + 28u], Ri = uint(P.y) * Rw + uint(P.x);
            const vec3 Eyw = vec3(TF(25u), TF(26u), TF(27u));
            const float Ah = push.AO != 0u ? GatherAO(P, W, H, Dist) : 1.0;
            // (C40: a skinned surface - XLION_SKIN_VOX's entry - is a point in open space, not one of the surfaces the store's faces stand for: read at a
            // walker's chest, a face of the cell (a wall's, the ground's, nobody's) gave black and bright blocks. Its rays (w 2) feed records of their own
            // (the faces' keys + 128) that only such surfaces read: the light of open space where movers were, which the rays - movers aren't in the BVH -
            // see the same whoever stands there. Fed to the shared store, they had blackened the whole voxel GI (2026-11-07))
            const bool bSkinRead = Ref == gi_probes_v;
            GIRefSample[2u * Ri] = vec4(S.P + Eyw, bSkinRead ? 2.0 : 1.0);
            GIRefSample[2u * Ri + 1u] = vec4(N, Ah);      // (w: this reader's AO - XLION_VOX_AO_BUCKET sorts its rays' samples by it)
            GIRefSample[2u * W * H] = vec4(TF(22u), TF(23u), TF(24u), TF(21u));
            g_ClipFloat = true;
            // (XLION_VOX_AO_BUCKET=<AO>, P1 #1 prototype: readers in a contact - their AO under it - read the records only such readers' rays fed, not the open floor's beside them; none yet: the shared ones)
            const float BucketAO = uintBitsToFloat(GITable[push.TableBase + 34u]);
            bool bPubMiss = true;
            if (GITable[push.TableBase + 37u] == 2u) { E = vec4(0.5, 0.5, 0.5, 1.0); bPubMiss = false; }      // (measurement: no level read at all)
            else if (GITable[push.TableBase + 37u] != 0u && !bSkinRead) { E = VoxReadPublished(S.P, N, bPubMiss); g_ClipTrust = 1.0; }
            if (bSkinRead)
            {
                const bool bProbes = S.Entry.x != 0u && GITable[push.TableBase + 30u] != 0u;
                vec3 Centre = S.P + Eyw;
                E = bProbes ? MoverProbesRead(S.P + Eyw, N, S.Entry.x - 1u, Centre) : vec4(0.0);      // (Milestone A: its own probes, else their last estimates <= 0.5 s)
                uint Path = E.w >= 1.0 ? 0u : (E.w > 0.0 ? 1u : 2u);
                if (E.w <= 0.0 && bProbes) E = VoxReadLevels(Centre - Eyw, N);      // (none, no history: the static store at the support's centre, by the pixel's normal - v3.1 #1)
                if (E.w <= 0.0) { Path = 3u; g_VoxBucket = 2u; E = VoxReadLevels(S.P, N); g_VoxBucket = 0u; }      // (no probes - XLION_MOVER_PROBES=0, or past the 64 movers: the +128 records, as before)
                if (GITable[push.TableBase + 30u] != 0u && GITable[push.TableBase + GITable[push.TableBase + 30u] + 5u] != 0u)      // (XLION_MOVER_DEBUG: the path taken)
                    E = vec4(Path == 0u ? vec3(0.1, 1.0, 0.1) : Path == 1u ? vec3(1.0, 1.0, 0.1) : Path == 2u ? vec3(0.1, 0.3, 1.0) : vec3(1.0, 0.1, 0.1), 1.0);
                bPubMiss = E.w <= 0.0;
            }      // (none yet - it just came: the surfaces' read until its rays come back)
#ifdef XLION_GI_BRICKS
            if (bPubMiss) E = VoxReadBricks(S.P, N);
#else
            if (bPubMiss) E = VoxReadLevels(S.P, N);
#endif
            if (GITable[push.TableBase + 37u] != 0u)
            {   // (statistics: pixels read published / fallen back)
                const uvec2 Pc = subgroupAdd(uvec2(bPubMiss ? 0u : 1u, bPubMiss ? 1u : 0u));
                if (subgroupElect()) { atomicAdd(PubU[2u * 4u], Pc.x); atomicAdd(PubU[2u * 4u + 1u], Pc.y); }
            }
            const float Wc = BucketAO > 0.0 ? smoothstep(BucketAO + 0.05, BucketAO - 0.1, Ah) : 0.0;      // (a soft hand-over by the AO: a hard one speckled - the AO a pixel at a time)
            if (Wc > 0.0) { g_VoxBucket = 1u; const float Tr = g_ClipTrust; const vec4 Ec = VoxReadLevels(S.P, N); g_VoxBucket = 0u; g_ClipTrust = Tr; if (Ec.w > 0.0) E.rgb = mix(E.rgb, Ec.rgb, Wc); }
            bWeak = E.w < 0.5;
            if ((GITable[push.TableBase + 38u] & 0x80000000u) != 0u) E.rgb = mix(vec3(1.0, 0.05, 0.05), vec3(0.05, 1.0, 0.05), g_ClipTrust);      // (XLION_VOX_TRUST_VIEW: the read's sample trust - red few samples, green full)
            E.w = 1.0;
            if (GITable[push.TableBase + 29u] == 0u) { g_GIRefl = VoxReflect(S.P, N, E.rgb); if ((GITable[push.TableBase + 32u] & 8u) != 0u) g_GIRefl = VoxDirsRead(S.P, N, g_GIRefl, P); }
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
        if (push.AO == 2u && E.w > 0.5) E.rgb += GatherSSGI(P, W, H);      // (the store's light kept for the free sectors, the screen's for the occluded ones)
        if (any(isnan(E.rgb)) || any(isinf(E.rgb))) E = vec4(0.0);     // (never a non-finite entry: the reprojection would carry it on)      // (a fresh read only: a reprojected one carries last frame's)
        if (GITable[push.TableBase + 29u] != 0u) g_GIRefl = vec4(0.0);
        const bool bE = E.w > 0.5, bR = g_GIRefl.w > 0.5;
        bKeep = !bE && !bR;           // (no GI here: the AO's own entry stays)
        // (never a non-finite or negative reflection, as E above: reprojection carries it on, black and blocky on the moto, Tomas 2026-11-04)
        vec3 Rr = bR && !any(isnan(g_GIRefl.rgb)) && !any(isinf(g_GIRefl.rgb)) ? max(g_GIRefl.rgb, vec3(0.0)) : vec3(0.0);
        // (the read along R over time, every pixel every frame: no tiles, no squares. Its bins' means move a little with each ray: on a still moto, 7 in 8 of the flicker went with the read)
        // the irradiance the same way (Tomas 2026-11-12: the GI used to lean on last frame's and flickered less; every tile has been fresh
        // each frame since 82c5988, with no history): this frame's a share of kGIScreenBlend over last frame's at the same surface
        if ((Hd.w & 1u) != 0u && (bR || bE))
        {
            const mat4 PrevW2C = mat4(uintBitsToFloat(GIScreen[1]), uintBitsToFloat(GIScreen[2]), uintBitsToFloat(GIScreen[3]), uintBitsToFloat(GIScreen[4]));
            const vec3 Prev    = S.P + uintBitsToFloat(GIScreen[5].xyz) + Moved;
            const vec4 C       = PrevW2C * vec4(Prev, 1.0);
            if (C.w > 1e-6)
            {
                // (C40 reopened, Tomas 2026-11-13: a mover takes no screen history at all - a skinned limb moves inside its instance, no reprojection follows it;
                // its own records are already a time mean. The history read bilinearly: HistoryBilinear - a partial footprint (an edge) takes its share less)
                if (!bMover)
                {
                    vec4 He, Hr; uint Age;
                    HistoryBilinear(C, W, H, length(Prev), N, He, Hr, Age);
                    if (bR && Hr.w > 0.0) Rr = mix(Rr, mix(Hr.rgb, Rr, 0.1), Hr.w);
                    if (bE && He.w > 0.0) E.rgb = mix(E.rgb, mix(He.rgb, E.rgb, kGIScreenBlend), He.w);      // (not weak entries': a hole carried on)
                }
            }
        }
        Out = uvec4(packHalf2x16(E.rg), packHalf2x16(vec2(E.b, Dist)), packHalf2x16(Rr.rg),
                    (packHalf2x16(vec2(Rr.b, 0.0)) & 0xFFFEu) | (bMover ? 1u : 0u) | Guide | (bE ? 1u << 30 : 0u) | (bR ? 1u << 31 : 0u) | (bE && bWeak ? 3u << 28 : 0u));     // (age 3, never a fresh one's: weak - a hole in the surfels' coverage, the Cornell cube's dashes)
    }
    if (bIn && !(bKeep && push.AO != 0u))
    {
        GIScreen[screen_header_v + uint(P.y) * W + uint(P.x)] = Out;
        // (the history: what stands still. Under a mover the static entry it hides is carried on from last frame - reprojected as a still point, so
        // the floor behind a walker keeps its settled light and is not read from nothing when the walker leaves (Tomas 2026-11-13); none: an empty entry)
        uvec4 Hist = Out;
        if (bMover)
        {
            Hist = uvec4(0u);
            if ((Hd.w & 1u) != 0u)
            {
                const mat4 PrevW2C = mat4(uintBitsToFloat(GIScreen[1]), uintBitsToFloat(GIScreen[2]), uintBitsToFloat(GIScreen[3]), uintBitsToFloat(GIScreen[4]));
                const vec4 C = PrevW2C * vec4(S.P + uintBitsToFloat(GIScreen[5].xyz), 1.0);
                const ivec2 Q = C.w > 1e-6 ? ivec2(floor((C.xy / C.w * 0.5 + 0.5) * vec2(float(W), float(H)))) : ivec2(-1);
                if (Q.x >= 0 && Q.y >= 0 && uint(Q.x) < W && uint(Q.y) < H)
                {
                    const uvec4 Rp = GIScreenPrev[screen_header_v + uint(Q.y) * W + uint(Q.x)];
                    if ((Rp.w & 1u) == 0u && (Rp.w & (3u << 30)) != 0u) Hist = Rp;
                }
            }
        }
        GIScreenHistory[screen_header_v + uint(P.y) * W + uint(P.x)] = Hist;
    }
#ifdef XLION_HEAT
    if (bIn) Heat[4u + uint(P.y) * W + uint(P.x)] = max(clock2x32ARB().x - HeatT0, 1u);
#endif

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
