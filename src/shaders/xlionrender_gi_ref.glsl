// The GI's ground truth for experiments (XLION_GI_REF, xLION_Docs gi_next_level_analysis.md): a brute path tracer through the GI's BVH, for
// the points the gather wrote last frame (no surfels read: the sun, lamps, sky and emission at each hit, Bounces diffuse bounces with the
// hits' albedo). Each pixel's irradiance averaged over the frames while its point stays where it is. Slow on purpose; never for a game.
// A pixel a lane; m_First: the next pixel (wraps); m_Epoch: the lights' version (another: the pixel starts over)
#ifdef GI_VOX_LANES
layout(local_size_x = GI_VOX_LANES) in;
#else
layout(local_size_x = 4) in;
#endif
//      // (a few lanes: lanes in a group pay for each other: 64 rays cost 0.9 ms in groups of 4, 2.9 in one of 64; 32768 cost 9 and 4: launching groups isn't free)

layout(std430, set = 0, binding = 5) readonly buffer samples_b { vec4 GIRefSample[]; };     // per pixel: its world point, w 1 (0: none); then its normal
layout(std430, set = 0, binding = 6)          buffer accum_b   { vec4 GIRefAccum[]; };      // per pixel 2: the sum (rgb) and count, its point
#ifdef GI_VOX_ONLY
// (XLION_VOX_RELIGHT, a prototype: rays' connections kept, 7 entries each: the point (w 1) and its normal; the ray; the hit (w: 1 a surface, 0 the sky, -1 a back face), its normal, albedo and emission. Relighting reads the light at a known hit, no traversal)
layout(std430, set = 0, binding = 7)          buffer conn_b    { uvec4 ConnHeader; vec4 Conn[]; };
layout(std430, set = 0, binding = 7)          buffer conn_u_b  { uint ConnU[]; };      // (the same memory: XLION_VOX_BRICKS's bitmaps, its word 3 the mark)
layout(set = 1, binding = 0) uniform sampler2D SceneHDR;      // (XLION_VOX_SEEN: last frame's lit image, what the surfaces were seen as: every light, shadow and texture on screen)
#endif
layout(std430, set = 0, binding = 6)          buffer accum_u_b { uvec4 GIRefAccumU[]; };    // (the same memory: a direction bin's fixed point sums, added atomically)
layout(push_constant) uniform push_t { int m_nPixels, m_First, m_Count, m_Frame, m_TlasRoot, m_PrimOffset, m_Bounces, m_Epoch; } Push;
#define XLION_GI_ALPHA_RAYS      // (the masked triangles' opacity: foliage_gi_occlusion_plan_2026-11-07.md)
#include "xlionrender_gi_pass_common.glsl"

uint RHash(uint x) { x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16; return x; }
uint g_Seed;
float Rnd() { g_Seed = RHash(g_Seed + 0x9E3779B9u); return float(g_Seed) / 4294967296.0; }
// (XLION_VOX_COHERENT, m_First's bit 3, measurement: the rays of nearby groups start from neighbouring pixels and share a direction in their surface's frame, another each frame: what does coherence buy?)
bool g_CohOn = false; float g_CohU = 0.0, g_CohV = 0.0;
vec3 CosineDir(in const vec3 N)
{
    const float u = g_CohOn ? fract(g_CohU + 0.02 * Rnd()) : Rnd(), v = g_CohOn ? fract(g_CohV + 0.02 * Rnd()) : Rnd(), r = sqrt(u), Phi = 6.2831853 * v;
    const vec3  T = normalize(cross(abs(N.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0), N)), B = cross(N, T);
    return normalize(T * (r * cos(Phi)) + B * (r * sin(Phi)) + N * sqrt(max(0.0, 1.0 - u)));
}

// The reflections' ground truth (XLION_REFL_GRADE=<roughness> with XLION_GI_REF, in the REFLECTIONS_MIRROR view): the first direction from the GGX lobe, the visible normals' (Heitz 2018), every
// surface that roughness over its flat normal. W: what this sample's Fresnel-free BRDF weight is (G2 / G1): the mean of the radiance by W is what the split sum's prefiltered Spec stands for, with the real lobe
vec3 SpecDir(in const vec3 N, in const vec3 V, in const float a, out float W)
{
    W = 0.0;
    const vec3 T = normalize(cross(abs(N.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0), N)), B = cross(N, T);
    const vec3 Ve = vec3(dot(V, T), dot(V, B), dot(V, N));
    if (Ve.z <= 1e-4) return N;
    const vec3  Vh = normalize(vec3(a * Ve.x, a * Ve.y, Ve.z));
    const float Lq = Vh.x * Vh.x + Vh.y * Vh.y;
    const vec3  T1 = Lq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(Lq) : vec3(1.0, 0.0, 0.0), T2 = cross(Vh, T1);
    const float u = Rnd(), v = Rnd(), r = sqrt(u), Phi = 6.2831853 * v, t1 = r * cos(Phi), s = 0.5 * (1.0 + Vh.z);
    const float t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * r * sin(Phi);
    const vec3  Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    const vec3  Ne = normalize(vec3(a * Nh.x, a * Nh.y, max(0.0, Nh.z)));
    const vec3  Le = reflect(-Ve, Ne);
    if (Le.z <= 1e-4) return N;
    const float LamV = 0.5 * (sqrt(1.0 + a * a * (1.0 - Ve.z * Ve.z) / (Ve.z * Ve.z)) - 1.0), LamL = 0.5 * (sqrt(1.0 + a * a * (1.0 - Le.z * Le.z) / (Le.z * Le.z)) - 1.0);
    W = (1.0 + LamV) / (1.0 + LamV + LamL);
    return T * Le.x + B * Le.y + N * Le.z;
}

// Experiment X1c (XLION_GI_VOX, Tomas's offset voxels lit by rays): light kept only where a surface is. Levels of 64^3 cells, 6 faces a cell, a face
// the sum of the samples (rgb) and their count, as floats (GIRefAccum). A ray goes from a real point of the screen (never inside a wall) into a cell
// around it, a face its normal looks toward, both picked by their read weights. m_Bounces: -1 this, -2 clear; m_Epoch: the samples a face keeps (negative: the hits feed too).
// The store is sparse (Tomas): a pool of faces by their world cell, nothing where no surface fed it; the light stays where it is as the eye moves.
// A face is three entries: its tag and the frame it last took a sample; the kept sum (rgb) and count, what the reads see; a short mean (rgb) and the samples' variance around it, which notice a change: when
// the short mean leaves the kept one by more than the noise says it can (Wicked's multiscale mean estimator, estimator::Add), the old samples go, in this face only.
// The header after the pixels (2 entries): the eye and level 0's cell
#include "xlionrender_vox_read.glsl"
// A sample into a level: a cell around the point and a face its normal looks toward, by the read's weights: it lands where the reads look for it
// The same rays, by where they went (the reflection's read, xlionrender_gi_gather_frag.glsl VoxReflect): a cell's face (no layers) keeps a plain mean per direction: around its normal,
// or along one of the other two axes, each way. No more rays: a place on a floor knows what its rays saw toward the street and toward the wall
vec3 g_VoxRayD = vec3(0.0);
bool g_VoxMissed = false;       // (this ray saw the sky. A bin keeps its sky apart, second entry: rgb sum, count: a direction near the surface takes less of it - VoxReflect)
bool g_HasSeen = false; vec3 g_SeenSample = vec3(0.0);      // (the bins take what a hit was seen as, the faces, the diffuse light, stay as the GI's lights give them)
void VoxStoreDir(in const int L, in const ivec3 Org, in const vec3 g, in const vec3 N, in const vec3 Sample0)
{
    const vec3 Sample = g_HasSeen ? g_SeenSample : Sample0;
    const vec3 An = abs(N);
    const int  a = An.x >= An.y && An.x >= An.z ? 0 : An.y >= An.z ? 1 : 2, u = (a + 1) % 3, v = (a + 2) % 3;
    const vec3 D = g_VoxRayD, Ad = abs(D);
    const uint b = Ad[a] >= Ad[u] && Ad[a] >= Ad[v] ? 4u : Ad[u] >= Ad[v] ? (D[u] > 0.0 ? 0u : 1u) : (D[v] > 0.0 ? 2u : 3u);
    const uint h = VoxHash(L, Org + ivec3(floor(g + 0.5)), 6u + uint(2 * a + (N[a] < 0.0 ? 1 : 0)) * 5u + b), e = VoxEntryOf(h);
    const float Tag = VoxTagOf(h), Now = float(uint(Push.m_Frame) & 0xFFFFFFu);
    // Atomic: lanes sharing a bin read, added and wrote over each other: samples lost, a bin's sum and its sky from different lanes. Its halving in a pass of its own (VoxMain, -4), one owner.
    // Entry e and e + 1: rgb by bin_scale_v, count; z of e + 2: 1, a bin
    const uint TagU = floatBitsToUint(Tag), Cur = GIRefAccumU[e + 1u].x;
    if (Cur != TagU)
    {
        if (Cur != 0u && abs(Now - GIRefAccum[VoxX2Of(e)].w) < 120.0) return;
        if (atomicCompSwap(GIRefAccumU[e + 1u].x, Cur, TagU) != Cur) return;
        GIRefAccumU[e] = uvec4(0u); GIRefAccumU[VoxX2Of(e)] = uvec4(0u); GIRefAccumU[e + 1u].yzw = uvec3(floatBitsToUint(1.0), 0u, 0u);
    }
    // (XLION_VOX_NO_BIN_PDF to turn it off, m_First's bit 2: a bin's mean over its directions, not by the cosine the rays came with: self normalized by 1 / pdf, clamped at a cosine of 0.1. The count in sixteenths; off, 16 a ray)
    const uint  w16 = (Push.m_First & 4) != 0 ? uint(16.0 / max(dot(N, g_VoxRayD), 0.1) + 0.5) : 16u;
    const uvec3 q = uvec3(clamp(Sample, vec3(0.0), vec3(4000.0)) * 16.0 * float(w16));
    atomicAdd(GIRefAccumU[e].x, q.x); atomicAdd(GIRefAccumU[e].y, q.y); atomicAdd(GIRefAccumU[e].z, q.z); atomicAdd(GIRefAccumU[e].w, w16);
    if (g_VoxMissed) { const uint e2 = VoxX2Of(e); atomicAdd(GIRefAccumU[e2].x, q.x); atomicAdd(GIRefAccumU[e2].y, q.y); atomicAdd(GIRefAccumU[e2].z, q.z); atomicAdd(GIRefAccumU[e + 1u].z, w16); }
    GIRefAccumU[VoxX2Of(e)].w = floatBitsToUint(Now);
}
// XLION_VOX_DIRS=<level>,<k> (m_First's bit 11; bits 12, 13: the level; bit 14: k 16, else 8): a prototype. Reflections' light in a coarse level, k x k directions a face (reflection_store_design_2026-11-06.md), the same rays
void VoxDirsAdd(in const vec3 Pw, in const vec3 N, in const float Cell0, in const vec3 Sample)
{
    const int   L = (Push.m_First >> 12) & 3; const uint k = (Push.m_First & 16384) != 0 ? 16u : 8u;
    const ivec3 q = ivec3(floor(Pw / (Cell0 * float(1 << L))));
    const vec3  An = abs(N);
    const int   a = An.x >= An.y && An.x >= An.z ? 0 : An.y >= An.z ? 1 : 2, u = (a + 1) % 3, v = (a + 2) % 3;
    const float sg = N[a] < 0.0 ? -1.0 : 1.0;
    const vec3  D = g_VoxRayD;
    const float s = abs(D[u]) + abs(D[v]) + max(D[a] * sg, 0.0) + 1e-6, x = D[u] / s, y = D[v] / s;
    const uint  i = min(uint((x + y + 1.0) * 0.5 * float(k)), k - 1u), j = min(uint((x - y + 1.0) * 0.5 * float(k)), k - 1u);
    const uint  h = VoxHash(L, q, 1000u + uint(2 * a + (N[a] < 0.0 ? 1 : 0)) * 256u + i * k + j), e = VoxEntryOf(h);
    const float Tag = VoxTagOf(h), Now = float(uint(Push.m_Frame) & 0xFFFFFFu);
    const uint  TagU = floatBitsToUint(Tag), Cur = GIRefAccumU[e + 1u].x;
    if (Cur != TagU)
    {
        if (Cur != 0u && abs(Now - GIRefAccum[VoxX2Of(e)].w) < 120.0) return;
        if (atomicCompSwap(GIRefAccumU[e + 1u].x, Cur, TagU) != Cur) return;
        GIRefAccumU[e] = uvec4(0u); GIRefAccumU[VoxX2Of(e)] = uvec4(0u); GIRefAccumU[e + 1u].yzw = uvec3(floatBitsToUint(1.0), 0u, 0u);
    }
    const uint  w16 = uint(16.0 / max(dot(N, g_VoxRayD), 0.1) + 0.5);
    const uvec3 c = uvec3(clamp(Sample, vec3(0.0), vec3(4000.0)) * 16.0 * float(w16));
    atomicAdd(GIRefAccumU[e].x, c.x); atomicAdd(GIRefAccumU[e].y, c.y); atomicAdd(GIRefAccumU[e].z, c.z); atomicAdd(GIRefAccumU[e].w, w16);
    GIRefAccumU[VoxX2Of(e)].w = floatBitsToUint(Now);
}
uint g_VoxStoreKind = 0u;      // (XLION_VOX_AO_BUCKET: 1 a contact's records - the faces' keys + 64; the direction bins stay shared)
#ifdef GI_VOX_ONLY
// (measurement, XLION_VOX_BRICKS - gi_store_bricks_plan_2026-11-08.md: each face a store writes marks a bit of three 2^21-bit maps, by the
// hash of its cell, of its 2 x 2 x 2 brick, of its 4 x 4 x 4 brick: distinct cells and bricks, so how full bricks would be)
const uint vox_brick_mark_v = 0x42524B53u;
void VoxBrickMark(in const int L, in const ivec3 q, in const uint Face)
{
    const uint R = 1u << 21;
    const uint h0 = VoxHash(L, q, Face) % R, h1 = VoxHash(L, q >> 1, Face + 16u) % R, h2 = VoxHash(L, q >> 2, Face + 32u) % R;
    atomicOr(ConnU[4u + (h0 >> 5)], 1u << (h0 & 31u));
    atomicOr(ConnU[4u + 65536u + (h1 >> 5)], 1u << (h1 & 31u));
    atomicOr(ConnU[4u + 131072u + (h2 >> 5)], 1u << (h2 & 31u));
}
#endif
const uint vox_busy_v = 0xFFFFFFFFu;      // (a face slot's tag while its new owner writes it: a NaN, equal to no tag)
void VoxStore(in const int L, in const ivec3 Org, in const vec3 g, in const vec3 N, in const vec3 Sample)
{
    VoxStoreDir(L, Org, g, N, Sample);
    const vec3  w = N * N;
    const float r = Rnd() * (w.x + w.y + w.z);
    const int   a = r < w.x ? 0 : r < w.x + w.y ? 1 : 2;
    vec3 gv = vec3(Org) + g; gv[a] *= vox_layers_v;
    const ivec3 i = ivec3(floor(gv)); const vec3 f = gv - vec3(i);
    const ivec3 q = i + ivec3(Rnd() < f.x ? 1 : 0, Rnd() < f.y ? 1 : 0, Rnd() < f.z ? 1 : 0);
    const uint  h = VoxHash(L, q, uint(2 * a + (N[a] < 0.0 ? 1 : 0)) + 64u * g_VoxStoreKind), e = VoxEntryOf(h);
    const float Tag = VoxTagOf(h), Now = float(uint(Push.m_Frame) & 0xFFFFFFu);
#ifdef GI_VOX_ONLY
    if (ConnU[3] == vox_brick_mark_v) VoxBrickMark(L, q, uint(2 * a + (N[a] < 0.0 ? 1 : 0)) + 64u * g_VoxStoreKind);
#endif
    vec4 v = GIRefAccum[e];
    const uvec4 dU = GIRefAccumU[e + 1u];
    const vec4 W2 = GIRefAccum[VoxX2Of(e)];
    vec4 m = vec4(W2.xyz, 0.0);
    const vec2 d = vec2(uintBitsToFloat(dU.x), W2.w);       // (the tag, the time)
    // (the lean moments, A15: running means of the luminance against the normal's two other components and of those components - the read fits the light's slope
    // across the normals that fed this face, so a sphere's bottom isn't read as its side. Halves in the spare words: e+1.w (variance, mean l.l), e+2.z (L.lu, L.lv; low bit
    // set: never the direction bins' 1.0 marker), e+2.w (lu, lv))
    vec2 Pv = unpackHalf2x16(dU.w), PL = unpackHalf2x16(dU.y & ~1u), Pl = unpackHalf2x16(dU.z);
    if (dU.x == vox_busy_v) return;      // (another lane is taking this slot over: its payload isn't whole yet)
    bool bTaken = false;
    if (d.x != Tag)
    {
        // someone else's: taken only if nobody has fed it for 2 seconds or so: a face in use keeps its slot and its history (a face that finds no room waits)
        if (d.x != 0.0 && abs(Now - d.y) < 120.0) return;
        // (C31: taken by compare-and-swap to a busy tag - one owner wins, the others leave; readers see the busy tag as a miss until the payload is whole)
        if (atomicCompSwap(GIRefAccumU[e + 1u].x, dU.x, vox_busy_v) != dU.x) return;
        bTaken = true;
        v = vec4(0.0); m = vec4(0.0); Pv = PL = Pl = vec2(0.0);
    }
    m.w = Pv.x;
    const float Ls = dot(Sample, vec3(0.2126, 0.7152, 0.0722));
    const int   ua = (a + 1) % 3, va = (a + 2) % 3;
    const vec2  Ln = vec2(N[ua], N[va]);
    if (v.w <= 0.0) { m = vec4(Sample, 0.0); Pv.y = dot(Ln, Ln); PL = Ls * Ln; Pl = Ln; }
    const float Ds = Ls - dot(m.rgb, vec3(0.2126, 0.7152, 0.0722));
    m.rgb += (Sample - m.rgb) * vox_short_v;
    m.w   += (Ds * Ds - m.w) * vox_short_v;
    if (v.w >= 64.0 && (Push.m_First & 1) != 0)       // (m_First: the lights changed lately)
    {
        const float Lk = dot(v.rgb, vec3(0.2126, 0.7152, 0.0722)) / v.w, Lm = dot(m.rgb, vec3(0.2126, 0.7152, 0.0722));
        const float Noise = sqrt(max(m.w, 0.0) * vox_short_v / (2.0 - vox_short_v));       // (the short mean's own standard deviation)
        if (abs(Lm - Lk) > vox_sigmas_v * Noise && abs(Lm - Lk) > 0.1 * Lk)
            v = vec4(m.rgb * 32.0, 32.0);                                                  // (it changed: the short mean is all it knows)
    }
    v += vec4(Sample, 1.0);
    if (v.w >= float(abs(Push.m_Epoch))) v *= 0.5 * float(abs(Push.m_Epoch)) / v.w;       // (ponytail: another lane's write to this face in this frame can be lost: samples, not light)
    const float Al = max(1.0 / v.w, 1.0 / 256.0);
    Pv.y += (dot(Ln, Ln) - Pv.y) * Al; PL += (clamp(Ls, 0.0, 60000.0) * Ln - PL) * Al; Pl += (Ln - Pl) * Al;
    // (C31: a taken slot publishes payload first, then its tag - a reader that sees the tag sees a whole record)
    GIRefAccum[e] = v;
    GIRefAccumU[VoxX2Of(e)] = uvec4(floatBitsToUint(m.rgb), floatBitsToUint(Now));
    if (bTaken) memoryBarrierBuffer();
    GIRefAccumU[e + 1u] = uvec4(floatBitsToUint(Tag), packHalf2x16(PL) | 1u, packHalf2x16(Pl), packHalf2x16(vec2(min(m.w, 60000.0), Pv.y)));
}
// The emitters' light by light sampling at a budget: EmitterIrradianceAt, but over a few of the table picked at random, not all of it per ray (Bistro's table: 3 ms a frame at any
// number of rays). The pick's weight scaled up by the table over the few. As noisy as it needs; a face averages it
vec3 VoxEmitters(in const vec3 P, in const vec3 N)
{
    const uint Tb = GD[14], nE = GD[15], Few = min(nE, 8u);
    float Q = 0.0; int Pick = -1;
    for (uint i = 0u; i < Few; ++i)
    {
        const uint k = nE <= 8u ? i : uint(Rnd() * float(nE)) % nE, e = Tb + 12u * k;
        const vec3 D = GV(e) - P;
        const float d2 = dot(D, D);
        if (d2 < 1e-10) continue;
        const vec3 Dn = D * (1.0 / sqrt(d2));
        const float cr = dot(N, Dn), ce = -dot(GV(e + 3u), Dn);
        if (cr <= 0.0 || ce <= 0.0) continue;
        const float A = GF(e + 6u), w = A * cr * ce / (d2 + A);
        if (w < 1e-5) continue;
        const vec3 Le = GV(e + 7u);
        const float q = w * (0.2126 * Le.x + 0.7152 * Le.y + 0.0722 * Le.z);
        if (q <= 0.0) continue;
        Q += q;
        if (Rnd() * Q < q) Pick = int(k);
    }
    if (Pick < 0) return vec3(0.0);
    const uint e = Tb + 12u * uint(Pick);
    const vec3 D = GV(e) - P;
    const float d = length(D);
    if (Occluded(P + N * 1e-3, D * (1.0 / d), d * 0.98)) return vec3(0.0);
    const vec3 Le = GV(e + 7u);
    return Le * (Q / (0.2126 * Le.x + 0.7152 * Le.y + 0.0722 * Le.z)) * (float(nE) / float(Few));
}
// The lights at a hit: the sun as Direct, of the lamps in reach one shadow ray, the lamp picked by what it would give (a shadow ray to every lamp was 2 of 5.75 ms at 16384 rays). As noisy as it needs; a face averages it
vec3 VoxDirect(in const vec3 P, in const vec3 N)
{
    g_DirectNoLamps = true;
    g_DirectNoSunShadow = (Push.m_First & 32768) != 0;      // (XLION_VOX_NO_SUN_SHADOW, measurement: the sun's shadow ray's share - S1's upper bound)
    vec3 E = Direct(P, N);
    const uint Lt = GD[2], nP = GD[3];
    float Q = 0.0, Dist = 0.0; vec3 Pick = vec3(0.0), Dir = vec3(0.0); float PickQ = 0.0;
    for (uint i = 0u; i < nP; ++i)
    {
        const uint b = Lt + 57u + 12u * i;
        const vec3 D = GV(b) - P;
        const float d = length(D), s = d / GF(b + 6u);
        if (s >= 1.0 || d < 1e-6) continue;
        const float c = dot(N, D) / d;
        if (c <= 0.0) continue;
        const float Co = GF(b + 10u);
        const float Cone = Co > -1.5 ? smoothstep(Co, max(GF(b + 11u), Co + 1e-6), dot(-D, GV(b + 7u)) / d) : 1.0;
        if (Cone <= 0.0) continue;
        const vec3 Li = GV(b + 3u) * ((1.0 - s * s * s * s) / (1.0 + 2.0 * s * s) * c * Cone);
        const float q = dot(Li, vec3(0.2126, 0.7152, 0.0722));
        if (q <= 0.0) continue;
        Q += q;
        if (Rnd() * Q < q) { Pick = Li; PickQ = q; Dir = D * (1.0 / d); Dist = d; }
    }
    if (Q > 0.0 && !Occluded(P + N * 1e-3, Dir, Dist - 2e-3)) E += Pick * (Q / PickQ);
    return E;
}
// The first bounce kept apart (Godot's direct light pass, Lumen's surface cache, Capsaicin's direct radiance; m_Bounces -3): a hit reads the direct light its surface was found to have, a face's mean (36 + the face),
// and one read in 8 casts its shadow rays again. The lights changed lately: every read casts them, the mean short. The sun's and the lamps' shadow rays were most of a ray's cost: light the voxels already knew
vec3 VoxDirectKept(in const vec3 P, in const vec3 N, in const vec4 Hd)
{
    int L; ivec3 Org; vec3 g;
    if (Push.m_Bounces != -3 || !VoxCell(P, Hd.xyz, Hd.w, 0, L, Org, g)) return VoxDirect(P, N);
    const vec3 An = abs(N);
    const int  a = An.x >= An.y && An.x >= An.z ? 0 : An.y >= An.z ? 1 : 2;
    vec3 gv = vec3(Org) + g; gv[a] *= vox_layers_v;
    const uint  h = VoxHash(L, ivec3(floor(gv + 0.5)), 36u + uint(2 * a + (N[a] < 0.0 ? 1 : 0))), e = VoxEntryOf(h);
    const float Tag = VoxTagOf(h), Now = float(uint(Push.m_Frame) & 0xFFFFFFu);
    const vec2  d = vec2(GIRefAccum[e + 1u].x, GIRefAccum[VoxX2Of(e)].w);
    const bool  bMine = d.x == Tag;
    vec4 c = bMine ? GIRefAccum[e] : vec4(0.0);
    if ((Push.m_First & 1) == 0 && c.w >= 1.0 && ((Push.m_First & 16) != 0 || Rnd() >= 0.125)) return c.rgb / c.w;      // (m_First's bit 4, XLION_VOX_HIT_CACHED, measurement: a hit with a kept direct never casts its shadow rays again: the best case of hits reading a cache)
    const vec3 E = VoxDirect(P, N);
    if (!bMine && d.x != 0.0 && abs(Now - d.y) < 120.0) return E;
    const uint T0 = floatBitsToUint(d.x);      // (C31: a takeover by compare-and-swap to the busy tag, the tag published last)
    if (T0 == vox_busy_v || (!bMine && atomicCompSwap(GIRefAccumU[e + 1u].x, T0, vox_busy_v) != T0)) return E;
    c += vec4(E, 1.0);
    if (c.w >= ((Push.m_First & 1) != 0 ? 4.0 : 32.0)) c *= 0.5;
    GIRefAccum[e] = c; GIRefAccum[VoxX2Of(e)] = vec4(0.0, 0.0, 0.0, Now);
    if (!bMine) memoryBarrierBuffer();
    GIRefAccum[e + 1u] = vec4(Tag, 0.0, 0.0, 0.0);
    return E;
}
// A cosine ray from a surface point: pi x what it sees (a sample of the point's indirect irradiance), and where it hit
int g_Ablate = 0;
float g_ConnKind = -1.0; vec3 g_ConnAlbedo = vec3(0.0), g_ConnEmission = vec3(0.0);
vec3 VoxRay(in const vec3 P, in const vec3 N, in const vec4 Hd, out vec3 Ph, out vec3 Nh, out bool bHit)
{
    bHit = false; Ph = vec3(0.0); Nh = vec3(0.0, 1.0, 0.0); g_HasSeen = false;
    // the emitters by light sampling (G5, as the surfels' pass: the table, a shadow ray), this point's own unless lighting shades it; an emission a ray hits is not counted again
    const vec3 Own = GD[15] > 0u && !EmitterDirectExternal() ? VoxEmitters(P, N) : vec3(0.0);
    const vec3 D = CosineDir(N);
    g_VoxRayD = D;
    const gi_hit H = GICast(P + N * 2e-3, D, 1e30);
    if (g_Ablate == 1) return vec3(0.0);
    g_VoxMissed = H.m_Inst < 0;
    g_ConnKind = -1.0;
    if (H.m_Inst < 0) { g_ConnKind = 0.0; return Own + SkyAt(D) * gi_pi_v; }
    const uint HR = InstRec(H.m_Inst), HM = MeshOf(HR);
    const uvec3 HT = MTri(HM, H.m_Tri);
    const float B0 = 1.0 - H.m_U - H.m_V;
    Nh = NormalToWorld(HR, MNrm(HM, HT.x) * B0 + MNrm(HM, HT.y) * H.m_U + MNrm(HM, HT.z) * H.m_V);
    if (dot(Nh, D) > 0.0) return Own;
    vec3 Albedo, Emission;
    SurfaceAt(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Albedo, Emission);
    Ph = ToWorld(HR, MPos(HM, HT.x) * B0 + MPos(HM, HT.y) * H.m_U + MPos(HM, HT.z) * H.m_V);
    bHit = true;
#ifdef GI_VOX_ONLY
    if ((Push.m_First & 128) != 0)
    {
        int Ls; ivec3 Os; vec3 gs;
        if (VoxCell(Ph, Hd.xyz, Hd.w, 0, Ls, Os, gs))
        {
            const vec3 An = abs(Nh);
            const int  a  = An.x >= An.y && An.x >= An.z ? 0 : An.y >= An.z ? 1 : 2;
            const uint h  = VoxHash(Ls, Os + ivec3(floor(gs + 0.5)), 42u + uint(2 * a + (Nh[a] < 0.0 ? 1 : 0))), e = VoxEntryOf(h);
            const vec4 c  = GIRefAccum[e];
            if (GIRefAccum[e + 1u].x == VoxTagOf(h) && c.w >= 1.0) { g_HasSeen = true; g_SeenSample = c.rgb / c.w * gi_pi_v; }
        }
    }
#endif
    if (g_Ablate == 2) return Albedo + Emission;
    g_ConnKind = 1.0; g_ConnAlbedo = Albedo; g_ConnEmission = Emission;
    vec3 In = VoxDirectKept(Ph, Nh, Hd);
    if (g_Ablate == 3) return In * Albedo;
    In += (Push.m_First & 2) != 0 ? vec3(0.0) : VoxRead(Ph, Nh, Hd.xyz, Hd.w);
    if (g_Ablate == 4) return In * Albedo;     // (m_First's bit 1, XLION_VOX_FIRST_ONLY: measurement: no bounce past the first, so nothing a sample reads is still converging)
    if (GD[15] > 0u) { In += VoxEmitters(Ph, Nh); Emission = vec3(0.0); }
    return Own + (Albedo * In * (1.0 / gi_pi_v) + Emission) * gi_pi_v;
}
void VoxMain()
{
    if (Push.m_Bounces == -2) { for (uint k = 0u; k < 16u; ++k) { const uint i = gl_GlobalInvocationID.x * 16u + k; if (int(i) < Push.m_Count) GIRefAccum[uint(Push.m_First) + i] = vec4(0.0); } return; }     // (a lane clears 16: not a million groups)
#ifdef GI_VOX_ONLY
    if (Push.m_Bounces == -7)         // (XLION_VOX_SEEN: the lit image into its surfaces' records, a mean of what they were seen as. m_Epoch: the half size width. ponytail: last frame's pixel without reprojection: a still camera's)
    {
        if (int(gl_GlobalInvocationID.x) >= Push.m_Count) return;
        const vec4 HdS = GIRefSample[2u * uint(Push.m_nPixels)];
        if (HdS.w <= 0.0) return;
        g_Seed = RHash(gl_GlobalInvocationID.x * 2654435761u + uint(Push.m_Frame) * 97u);
        const uint px = uint(Rnd() * float(Push.m_nPixels)) % uint(Push.m_nPixels), W1 = uint(Push.m_Epoch);
        const vec4 Ps = GIRefSample[2u * px];
        if (Ps.w < 0.5) return;
        const vec3 Nk = GIRefSample[2u * px + 1u].xyz;
        int Ls; ivec3 Os; vec3 gs;
        if (!VoxCell(Ps.xyz, HdS.xyz, HdS.w, 0, Ls, Os, gs)) return;
        const vec3 Lit = texelFetch(SceneHDR, ivec2(int(px % W1) * 2 + 1, int(px / W1) * 2 + 1), 0).rgb;
        if (any(isnan(Lit)) || any(isinf(Lit))) return;
        const vec3 An = abs(Nk);
        const int  a  = An.x >= An.y && An.x >= An.z ? 0 : An.y >= An.z ? 1 : 2;
        const uint h  = VoxHash(Ls, Os + ivec3(floor(gs + 0.5)), 42u + uint(2 * a + (Nk[a] < 0.0 ? 1 : 0))), e = VoxEntryOf(h);
        const float Tag = VoxTagOf(h), Now = float(uint(Push.m_Frame) & 0xFFFFFFu);
        const vec2 d = vec2(GIRefAccum[e + 1u].x, GIRefAccum[VoxX2Of(e)].w);
        vec4 c = d.x == Tag ? GIRefAccum[e] : vec4(0.0);
        if (d.x != Tag && d.x != 0.0 && abs(Now - d.y) < 120.0) return;
        const uint T0 = floatBitsToUint(d.x);      // (C31: as VoxStore - taken by compare-and-swap, the tag published last)
        if (T0 == vox_busy_v || (d.x != Tag && atomicCompSwap(GIRefAccumU[e + 1u].x, T0, vox_busy_v) != T0)) return;
        c += vec4(min(Lit, vec3(4000.0)), 1.0);
        if (c.w >= 64.0) c *= 0.5;
        GIRefAccum[e] = c; GIRefAccum[VoxX2Of(e)] = vec4(0.0, 0.0, 0.0, Now);
        if (d.x != Tag) memoryBarrierBuffer();
        GIRefAccum[e + 1u] = vec4(Tag, 0.0, 0.0, 0.0);
        return;
    }
    if (Push.m_Bounces == -6) { if (gl_GlobalInvocationID.x == 0u) ConnHeader = uvec4(0u); for (uint k = 0u; k < 16u; ++k) { const uint i = gl_GlobalInvocationID.x * 16u + k; if (i < uint(Conn.length())) Conn[i] = vec4(0.0); } return; }
#endif
    if (Push.m_Bounces == -4)         // (the bins' halving, a slice of the slots a frame: nothing else writes them now)
    {
        for (uint k = 0u; k < 16u; ++k)
        {
            const uint i = gl_GlobalInvocationID.x * 16u + k;
            if (int(i) >= Push.m_Count) return;
            const uint s0 = uint(Push.m_First) + i, e = 3u * s0 + (s0 & 1u);
            if (GIRefAccum[e + 1u].y != 1.0) continue;
            const uvec4 c = GIRefAccumU[e];
            if (c.w < uint(Push.m_Epoch) * 16u) continue;
            const float s = 8.0 * float(Push.m_Epoch) / float(c.w);
            const uint e2 = VoxX2Of(e);
            GIRefAccumU[e] = uvec4(vec4(c) * s); GIRefAccumU[e2].xyz = uvec3(vec3(GIRefAccumU[e2].xyz) * s); GIRefAccumU[e + 1u].z = uint(float(GIRefAccumU[e + 1u].z) * s);
        }
        return;
    }
    if (int(gl_GlobalInvocationID.x) >= Push.m_Count) return;
    g_GIBvhTlasRoot = Push.m_TlasRoot; g_GIBvhPrimOffset = Push.m_PrimOffset; g_GIBvhSeeThrough = true;
    const vec4 Hd = GIRefSample[2u * uint(Push.m_nPixels)];
    if (Hd.w <= 0.0) return;
    g_Ablate = (Push.m_First >> 8) & 7;
    g_Seed = RHash(gl_GlobalInvocationID.x * 2654435761u + uint(Push.m_Frame) * 97u);
#ifdef GI_VOX_ONLY
    if ((Push.m_First & 64) != 0)      // (relight: a known connection with the light of now)
    {
        const uint nConn = uint(Conn.length()) / 7u, k = (uint(Push.m_Frame) * uint(Push.m_Count) + gl_GlobalInvocationID.x) % nConn;
        const vec4 C0 = Conn[7u * k];
        if (C0.w < 0.5) return;
        const vec3 N1 = Conn[7u * k + 1u].xyz, D1 = Conn[7u * k + 2u].xyz;
        const vec4 H3 = Conn[7u * k + 3u];
        g_VoxRayD = D1; g_VoxMissed = H3.w == 0.0;
        vec3 Sample = vec3(0.0);
        if (H3.w == 0.0) Sample = SkyAt(D1) * gi_pi_v;
        else if (H3.w > 0.5)
        {
            const vec3 Nh1 = Conn[7u * k + 4u].xyz, Al = Conn[7u * k + 5u].xyz, Em = Conn[7u * k + 6u].xyz;
            vec3 In = VoxDirectKept(H3.xyz, Nh1, Hd) + VoxRead(H3.xyz, Nh1, Hd.xyz, Hd.w);
            Sample = (Al * In * (1.0 / gi_pi_v) + Em) * gi_pi_v;
        }
        int L1; ivec3 O1; vec3 g1;
        if (!any(isnan(Sample)) && all(lessThan(Sample, vec3(1e6))) && VoxCell(C0.xyz, Hd.xyz, Hd.w, 0, L1, O1, g1))
        {
            VoxStore(L1, O1, g1, N1, Sample);
            int Lp1; ivec3 Op1; vec3 gp1;
            if (VoxCell(C0.xyz, Hd.xyz, Hd.w, L1 + 1, Lp1, Op1, gp1)) VoxStore(Lp1, Op1, gp1, N1, Sample);
        }
        return;
    }
#endif
    // Of a few pixels, the one whose cell knows least: the rays go where they are needed - the cells a moving eye just brought in - and not, as the screen's area would have it, to the near floor
    vec3 P = vec3(0.0), N = vec3(0.0, 1.0, 0.0);
    int L = 0; ivec3 Org = ivec3(0); vec3 g = vec3(0.0);
    float Least = 1e30, OriginAO = 1.0;
    const bool bCoh = (Push.m_First & 8) != 0;
    if (bCoh)
    {
        g_CohOn = true;
        const uint Run = RHash(gl_WorkGroupID.x * 2654435761u + uint(Push.m_Frame) * 97u);      // (a group of 64 shares a direction: more shared lost directions: 21.7 against 17.9 at roughness 0.3)
        g_CohU = float(Run & 0xFFFFu) / 65536.0; g_CohV = float(Run >> 16u) / 65536.0;
    }
    for (int k = 0; k < (bCoh ? 1 : 4); ++k)
    {
        const uint  px = bCoh ? (RHash(uint(Push.m_Frame)) + gl_GlobalInvocationID.x * 7u) % uint(Push.m_nPixels) : uint(Rnd() * float(Push.m_nPixels)) % uint(Push.m_nPixels);
        const vec4  Ps = GIRefSample[2u * px];
        if (Ps.w < 0.5) continue;
        const vec3  Nk = GIRefSample[2u * px + 1u].xyz;
        int Lk; ivec3 Ok; vec3 gk;
        if (!VoxCell(Ps.xyz, Hd.xyz, Hd.w, 0, Lk, Ok, gk)) continue;
        const int   Ax = abs(Nk.x) >= abs(Nk.y) && abs(Nk.x) >= abs(Nk.z) ? 0 : abs(Nk.y) >= abs(Nk.z) ? 1 : 2;
        vec3 gq = vec3(Ok) + gk; gq[Ax] *= vox_layers_v;
        const float n  = VoxFace(Lk, ivec3(floor(gq + 0.5)), uint(2 * Ax + (Nk[Ax] < 0.0 ? 1 : 0))).w;
        if (n < Least) { Least = n; P = Ps.xyz; N = Nk; L = Lk; Org = Ok; g = gk; OriginAO = GIRefSample[2u * px + 1u].w; }
    }
    if (Least >= 1e30) return;
    const float BucketAO = uintBitsToFloat(uint(Push.m_First >> 16) << 16);      // (XLION_VOX_AO_BUCKET: the AO in m_First's top 16 bits, as a float's)
    vec3 Ph, Nh; bool bHit;
    // (one copy of VoxRay: the shader's compiled size goes with its copies. The second ray: the surface the first hit, seen yet or not: a sample of that surface's own
    // indirect light; off screen surfaces are lit before the eye turns to them. A group in four, all its lanes: one lane firing it makes its whole group wait)
    for (int r = 0; r < 2; ++r)
    {
        const vec3 Sample = VoxRay(P, N, Hd, Ph, Nh, bHit);
        if (g_Ablate >= 1 && g_Ablate <= 4) { if (any(isnan(Sample))) GIRefAccumU[0].w += 0u; return; }
#ifdef GI_VOX_ONLY
        if (r == 0 && (Push.m_First & 32) != 0)      // (kept for relighting: the ring's oldest goes)
        {
            const uint k = atomicAdd(ConnHeader.x, 1u) % (uint(Conn.length()) / 7u);
            Conn[7u * k] = vec4(P, 1.0); Conn[7u * k + 1u] = vec4(N, 0.0); Conn[7u * k + 2u] = vec4(g_VoxRayD, 0.0); Conn[7u * k + 3u] = vec4(Ph, g_ConnKind);
            Conn[7u * k + 4u] = vec4(Nh, 0.0); Conn[7u * k + 5u] = vec4(g_ConnAlbedo, 0.0); Conn[7u * k + 6u] = vec4(g_ConnEmission, 0.0);
        }
#endif
        if (!any(isnan(Sample)) && all(lessThan(Sample, vec3(1e6))))
        {
            g_VoxStoreKind = r == 0 && BucketAO > 0.0 && OriginAO < BucketAO ? 1u : 0u;
            VoxStore(L, Org, g, N, Sample);
            if ((Push.m_First & 2048) != 0) VoxDirsAdd(P, N, Hd.w, g_HasSeen ? g_SeenSample : Sample);
            int Lp; ivec3 Op; vec3 gp;
            if (r == 0 && VoxCell(P, Hd.xyz, Hd.w, L + 1, Lp, Op, gp)) VoxStore(Lp, Op, gp, N, Sample);
        }
        if (g_Ablate == 5 || r == 1 || Push.m_Epoch >= 0 || !bHit || (RHash(gl_WorkGroupID.x * 2654435761u + uint(Push.m_Frame) * 97u) & 255u) >= 64u || !VoxCell(Ph, Hd.xyz, Hd.w, 0, L, Org, g)) break;
        P = Ph; N = Nh;
    }
}

void main()
{
#ifdef GI_VOX_ONLY
    VoxMain();
#else
    if (int(gl_GlobalInvocationID.x) >= Push.m_Count) return;
    g_GIBvhTlasRoot = Push.m_TlasRoot; g_GIBvhPrimOffset = Push.m_PrimOffset; g_GIBvhSeeThrough = true;
    const uint i = (uint(Push.m_First) + gl_GlobalInvocationID.x) % uint(Push.m_nPixels);
    const vec4 Ps = GIRefSample[2u * i];
    if (Ps.w < 0.5) return;
    const vec3 N0 = GIRefSample[2u * i + 1u].xyz;
    g_Seed = RHash(i * 2654435761u + uint(Push.m_Frame) * 97u);
    vec3 P = Ps.xyz, N = N0, Th = vec3(1.0), Sum = vec3(0.0);
    const vec4  Grade = GIRefSample[2u * uint(Push.m_nPixels)];      // (after the pixels: the eye, w the roughness graded; negative: the irradiance)
    const bool  bSpec = Grade.w >= 0.0;
    float SpecW = 1.0;
    if (!bSpec && GD[15] > 0u && !EmitterDirectExternal()) Sum += EmitterIrradianceAt(P, N, g_Seed) * (1.0 / gi_pi_v);      // (the point's own: lighting doesn't shade them, the GI shows them)
    // (Russian roulette past m_Bounces, as Unreal's path tracer: a fixed count cut the light that took more bounces off - a dark bias in closed places.
    // Kept with the path's throughput, at most .95, the survivors weighed up by it; 16 bounces at most)
    for (int b = 0; b < 16; ++b)
    {
        if (Push.m_Bounces >= 100) { if (b >= Push.m_Bounces - 100) break; }      // (XLION_GI_REF=<paths>,1<nn>, measurement: a hard cap of nn bounces, no roulette - the voxels' XLION_VOX_FIRST_ONLY's ground truth)
        else if (b > Push.m_Bounces) { const float q = clamp(max(Th.r, max(Th.g, Th.b)), 0.05, 0.95); if (Rnd() >= q) break; Th /= q; }
        const vec3 D = bSpec && b == 0 ? SpecDir(N, normalize(Grade.xyz - P), max(fract(Grade.w) * fract(Grade.w), 1e-3), SpecW) : CosineDir(N);
        if (SpecW <= 0.0) return;
        if (Grade.w >= 2.0) SpecW = 1.0;      // (XLION_REFL_REF_NO_WEIGHT, measurement: the lobe's samples unweighted, as the reflection rays' records are)
        const gi_hit H = GICast(P + N * 2e-3, D, 1e30);
        if (H.m_Inst < 0) { Sum += Th * SkyAt(D); break; }
        const uint HR = InstRec(H.m_Inst), HM = MeshOf(HR);
        const uvec3 HT = MTri(HM, H.m_Tri);
        const float B0 = 1.0 - H.m_U - H.m_V;
        const vec3 Nh = NormalToWorld(HR, MNrm(HM, HT.x) * B0 + MNrm(HM, HT.y) * H.m_U + MNrm(HM, HT.z) * H.m_V);
        if (dot(Nh, D) > 0.0) break;          // (a back face: no light, as the CPU reference and the probes)
        vec3 Albedo, Emission;
        SurfaceAt(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Albedo, Emission);
        const vec3 Ph = ToWorld(HR, MPos(HM, HT.x) * B0 + MPos(HM, HT.y) * H.m_U + MPos(HM, HT.z) * H.m_V);
        vec3 In = Direct(Ph, Nh);
        if (GD[15] > 0u) { In += EmitterIrradianceAt(Ph, Nh, g_Seed); Emission = vec3(0.0); g_Seed = RHash(g_Seed); }        // (G5: the emitters by light sampling; not counted again when a ray hits one)
        Sum += Th * (Albedo * In * (1.0 / gi_pi_v) + Emission);
        Th *= Albedo;
        P = Ph; N = Nh;
    }
    // irradiance: pi times the mean radiance of cosine-weighted rays. A pixel whose point moved starts over
    vec4 A = GIRefAccum[2u * i];
    const vec4 Pa = GIRefAccum[2u * i + 1u];
    if (Pa.w != float(Push.m_Epoch) || dot(Pa.xyz - Ps.xyz, Pa.xyz - Ps.xyz) > 1e-4) { A = vec4(0.0); GIRefAccum[2u * i + 1u] = vec4(Ps.xyz, float(Push.m_Epoch)); }
    if (all(lessThan(Sum, vec3(1e6))) && !any(isnan(Sum))) A += vec4(Sum * gi_pi_v * SpecW, SpecW);
    GIRefAccum[2u * i] = A;
#endif
}
