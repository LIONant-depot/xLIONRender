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
// (Milestone A, the movers' probes - milestone_a_impl_2026-11-13.md: this frame's lattices, written by the CPU, a region a frame parity at Push.m_First:
// [0] (movers, alpha, the validity's step, 0); then per mover (origin xyz, the x/z spacing), (the y spacing, reset 1, 0, 0). The probes themselves:
// the store's tail, GIRefAccum[vox_probe_base_v + 4 (12 mover + probe) + 0..3]: SH L1 R, G, B, (validity, frames, 0, 0))
layout(std430, set = 0, binding = 8) readonly buffer mover_b { vec4 Mover[]; };
#endif
layout(std430, set = 0, binding = 6)          buffer accum_u_b { uvec4 GIRefAccumU[]; };    // (the same memory: a direction bin's fixed point sums, added atomically)
layout(std430, set = 0, binding = 6)          buffer accum_w_b { uint GIRefAccumW[]; };     // (the same memory by words: the faces' freeze states)
layout(push_constant) uniform push_t { int m_nPixels, m_First, m_Count, m_Frame, m_TlasRoot, m_PrimOffset, m_Bounces, m_Epoch; } Push;
#ifndef VOX_AMORTIZE
#define VOX_AMORTIZE 0.0      // (1.0 tried 2026-11-12: lit at 20 s 26.8 -> 29.1 %, frame 9.0 -> 8.8 ms - not kept; the store's far levels already need few rays)
#endif
#ifndef VOX_PICKS
#define VOX_PICKS 4       // (the pixels a ray chooses its least-known face from; 16 tried 2026-11-12: no fewer flashes at thin edges, 10 s resolve 34.5 -> 41.1 % off)
#endif
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
#include "xlionrender_sh_l1.glsl"
// A sample into a level: a cell around the point and a face its normal looks toward, by the read's weights: it lands where the reads look for it
// The same rays, by where they went (the reflection's read, xlionrender_gi_gather_frag.glsl VoxReflect): a cell's face (no layers) keeps a plain mean per direction: around its normal,
// or along one of the other two axes, each way. No more rays: a place on a floor knows what its rays saw toward the street and toward the wall
// (settle phase 1, 2026-11-13 - gi_settle_and_movers_plan_2026-11-13.md: a slot is in use while its owner was WRITTEN (X2.w) or SEEN (X2.z: the
// rays' pick marks the faces of the points the camera sees) in the last 120 frames. A converged face takes no rays, so by writes alone it was
// taken by another cell after 2 s and restarted from one sample: near, settled faces churned for ever. A direction bin (X1.y 1.0) keeps sky
// sums in X2.xyz: its last write only)
bool VoxSlotInUse(in const uint e, in const float Now)
{
    const vec4 X2 = GIRefAccum[VoxX2Of(e)];
    bool bKeep = true;
#ifdef GI_VOX_ONLY
    bKeep = ConnU[2] != 0x44524654u || (ConnU[12] & 1u) == 0u;      // (measurement, XLION_VOX_NO_KEEPALIVE with XLION_VOX_DRIFT: the old rule, writes only)
#endif
    return abs(Now - X2.w) < 120.0 || (bKeep && GIRefAccum[e + 1u].y != 1.0 && abs(Now - X2.z) < 120.0);
}
vec3 g_VoxRayD = vec3(0.0);
bool g_VoxMissed = false;       // (this ray saw the sky. A bin keeps its sky apart, second entry: rgb sum, count: a direction near the surface takes less of it - VoxReflect)
bool g_DriftProbe = false;      // (Y4: this ray a probe of the whole store - one random pixel, no pick, no inconsistency gate: only probes feed the drift)
bool g_HasSeen = false; vec3 g_SeenSample = vec3(0.0);      // (the bins take what a hit was seen as, the faces, the diffuse light, stay as the GI's lights give them)
void VoxStoreDir(in const int L, in const ivec3 Org, in const vec3 g, in const vec3 N, in const vec3 Sample0)
{
    const vec3 Sample = g_HasSeen ? g_SeenSample : Sample0;
    const vec3 An = abs(N);
    const int  a = An.x >= An.y && An.x >= An.z ? 0 : An.y >= An.z ? 1 : 2, u = (a + 1) % 3, v = (a + 2) % 3;
    const vec3 D = g_VoxRayD, Ad = abs(D);
    const uint b = Ad[a] >= Ad[u] && Ad[a] >= Ad[v] ? 4u : Ad[u] >= Ad[v] ? (D[u] > 0.0 ? 0u : 1u) : (D[v] > 0.0 ? 2u : 3u);
    const uvec2 h = VoxHash(L, Org + ivec3(floor(g + 0.5)), 6u + uint(2 * a + (N[a] < 0.0 ? 1 : 0)) * 5u + b); const uint e = VoxEntryOf(h);
    const float Tag = VoxTagOf(h), Now = float(uint(Push.m_Frame) & 0xFFFFFFu);
    // Atomic: lanes sharing a bin read, added and wrote over each other: samples lost, a bin's sum and its sky from different lanes. Its halving in a pass of its own (VoxMain, -4), one owner.
    // Entry e and e + 1: rgb by bin_scale_v, count; z of e + 2: 1, a bin
    const uint TagU = floatBitsToUint(Tag), Cur = GIRefAccumU[e + 1u].x;
    if (Cur != TagU)
    {
        if (Cur != 0u && VoxSlotInUse(e, Now)) return;
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
    const uvec2  h = VoxHash(L, q, 1000u + uint(2 * a + (N[a] < 0.0 ? 1 : 0)) * 256u + i * k + j); const uint e = VoxEntryOf(h);
    const float Tag = VoxTagOf(h), Now = float(uint(Push.m_Frame) & 0xFFFFFFu);
    const uint  TagU = floatBitsToUint(Tag), Cur = GIRefAccumU[e + 1u].x;
    if (Cur != TagU)
    {
        if (Cur != 0u && VoxSlotInUse(e, Now)) return;
        if (atomicCompSwap(GIRefAccumU[e + 1u].x, Cur, TagU) != Cur) return;
        GIRefAccumU[e] = uvec4(0u); GIRefAccumU[VoxX2Of(e)] = uvec4(0u); GIRefAccumU[e + 1u].yzw = uvec3(floatBitsToUint(1.0), 0u, 0u);
    }
    const uint  w16 = uint(16.0 / max(dot(N, g_VoxRayD), 0.1) + 0.5);
    const uvec3 c = uvec3(clamp(Sample, vec3(0.0), vec3(4000.0)) * 16.0 * float(w16));
    atomicAdd(GIRefAccumU[e].x, c.x); atomicAdd(GIRefAccumU[e].y, c.y); atomicAdd(GIRefAccumU[e].z, c.z); atomicAdd(GIRefAccumU[e].w, w16);
    GIRefAccumU[VoxX2Of(e)].w = floatBitsToUint(Now);
}
uint g_VoxStoreKind = 0u;      // (XLION_VOX_AO_BUCKET: 1 a contact's records - the faces' keys + 64; the direction bins stay shared. 2: a skinned surface's - open space, + 128: C40)
#ifdef GI_VOX_ONLY
// (measurement, XLION_VOX_BRICKS - gi_store_bricks_plan_2026-11-08.md: each face a store writes marks a bit of three 2^21-bit maps, by the
// hash of its cell, of its 2 x 2 x 2 brick, of its 4 x 4 x 4 brick: distinct cells and bricks, so how full bricks would be)
const uint vox_brick_mark_v = 0x42524B53u;
void VoxBrickMark(in const int L, in const ivec3 q, in const uint Face)
{
    const uint R = 1u << 21;
    const uint h0 = VoxHash(L, q, Face).x % R, h1 = VoxHash(L, q >> 1, Face + 16u).x % R, h2 = VoxHash(L, q >> 2, Face + 32u).x % R;
    atomicOr(ConnU[4u + (h0 >> 5)], 1u << (h0 & 31u));
    atomicOr(ConnU[4u + 65536u + (h1 >> 5)], 1u << (h1 & 31u));
    atomicOr(ConnU[4u + 131072u + (h2 >> 5)], 1u << (h2 & 31u));
}
// (measurement, XLION_VOX_DRIFT - gi_settled_stop_design_2026-11-12.md: the store-wide signed drift, running totals in 1/4096 fixed point (1/16 truncated the many small negative deviations: a false +.115) the CPU
// reads as differences: the samples' luminance above and below their face's short mean, the short means, the stores. Noise cancels, propagation adds up)
const uint vox_drift_mark_v = 0x44524654u;
const uint vox_glide_mark_v = 0x474C4944u;     // (XLION_VOX_GLIDE: ConnU[1])
void VoxDriftAdd(in const float Ds, in const float Lm)
{
    const float d = clamp(Ds, -64.0, 64.0);
    if (d > 0.0) atomicAdd(ConnU[4], uint(d * 4096.0)); else atomicAdd(ConnU[5], uint(-d * 4096.0));
    atomicAdd(ConnU[6], uint(clamp(Lm, 0.0, 64.0) * 4096.0));
    atomicAdd(ConnU[7], 1u);
}
#endif
const uint vox_busy_v = 0xFFFFFFFFu;      // (a face slot's tag while its new owner writes it: a NaN, equal to no tag)
// Per-face freeze (XLION_VOX_FREEZE, static_settle_plan v4, 2026-11-14 - replaces the global settle stop): a word a slot after the movers' probes and their scratch
// (gpu_trace::vox_freeze_base_v): bits 0-30 the samples the face took since the lights last stood still (the CPU clears the words then), bit 31 frozen.
// Frozen: read-only, no ray picks it, the camera never wakes it - only a light change does (while the lights move nothing is frozen or counted).
// A face freezes after the budget (m_First bits 16-30, x 16) of fresh samples, or after an eighth of it when it already holds half its window and its
// short and long means agree (the old freeze's rule). ponytail: one global light gate, no dependency wake - a frozen face misses later propagation; wake by sources later
const uint vox_freeze_base_v = vox_probe_base_v + 4u * 12u * vox_movers_v * 33u;      // (uvec4 index: gpu_trace::vox_freeze_base_v)
const uint vox_frozen_v = 0x80000000u;
// (m_Frame's top 8 bits: bit 31 on, 24-30 the budget / 16 - the time and seeds use its low 24. Not m_First's top: XLION_VOX_AO_BUCKET, on by default,
// keeps its AO there - the first build put the freeze there and it never ran; neither did the old XLION_VOX_FREEZE)
bool  VoxFreezeOn(void) { return (uint(Push.m_Frame) & 0x80000000u) != 0u && (Push.m_First & 1) == 0; }      // (on, and the lights still)
float VoxFreezeBudget(void) { return float((uint(Push.m_Frame) >> 24u) & 0x1Fu) * 64.0; }      // (bits 24-28, x 64; bit 29: parents count and keep the rays coming; bit 30: the early freeze of calm faces)
bool  VoxFreezeParents(void) { return (uint(Push.m_Frame) & 0x20000000u) != 0u; }
uint  VoxFreezeWord(in const uint h) { return 4u * vox_freeze_base_v + h % vox_slots_v; }
bool  g_VoxParentStore = false;      // (a coarser level's store from a finer face's ray: not counted - the plan's 'parent writes are not evidence')
void VoxStore(in const int L, in const ivec3 Org, in const vec3 g, in const vec3 N, in const vec3 SampleIn)
{
    vec3 Sample = SampleIn;
    const bool bFreeze = VoxFreezeOn() && g_VoxStoreKind != 2u;      // (a skinned surface's records move with it: never frozen)
    if (!bFreeze && g_VoxStoreKind != 2u) VoxStoreDir(L, Org, g, N, Sample);      // (a mover's open space point: no reflection bins, they are the surfaces')
    const vec3  w = N * N;
    const float r = Rnd() * (w.x + w.y + w.z);
    const int   a = r < w.x ? 0 : r < w.x + w.y ? 1 : 2;
    vec3 gv = vec3(Org) + g; gv[a] *= vox_layers_v;
    const ivec3 i = ivec3(floor(gv)); const vec3 f = gv - vec3(i);
    const ivec3 q = i + ivec3(Rnd() < f.x ? 1 : 0, Rnd() < f.y ? 1 : 0, Rnd() < f.z ? 1 : 0);
    const uvec2  h = VoxHash(L, q, uint(2 * a + (N[a] < 0.0 ? 1 : 0)) + 64u * g_VoxStoreKind); const uint e = VoxEntryOf(h);
    const float Tag = VoxTagOf(h), Now = float(uint(Push.m_Frame) & 0xFFFFFFu);
#ifdef GI_VOX_ONLY
    if (ConnU[3] == vox_brick_mark_v) VoxBrickMark(L, q, uint(2 * a + (N[a] < 0.0 ? 1 : 0)) + 64u * g_VoxStoreKind);
#endif
    vec4 v = GIRefAccum[e];
    const uvec4 dU = GIRefAccumU[e + 1u];
    const vec4 W2 = GIRefAccum[VoxX2Of(e)];
    // (X2: the short mean as halves (r g, b + the inconsistency), a spare word, the time - kajiya's / WickedEngine's estimator: how far the long mean
    // is from the short one in the samples' own deviations, smoothed; the rays go where it is high, none where a face is resolved - Tomas 2026-11-12)
    const vec2 W2b = unpackHalf2x16(floatBitsToUint(W2.y));
    vec4 m = vec4(unpackHalf2x16(floatBitsToUint(W2.x)), W2b.x, 0.0);
    float Incon = W2b.y;
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
        if (d.x != 0.0 && VoxSlotInUse(e, Now))
        {
#ifdef GI_VOX_ONLY
            if (ConnU[2] == vox_drift_mark_v) atomicAdd(ConnU[9], 1u);      // (step B, XLION_VOX_DRIFT: a face refused - its slot in use by another)
#endif
            return;
        }
        // (C31: taken by compare-and-swap to a busy tag - one owner wins, the others leave; readers see the busy tag as a miss until the payload is whole)
        if (atomicCompSwap(GIRefAccumU[e + 1u].x, dU.x, vox_busy_v) != dU.x) return;
        bTaken = true;
#ifdef GI_VOX_ONLY
        if (ConnU[2] == vox_drift_mark_v) atomicAdd(ConnU[d.x != 0.0 ? 8 : 10], 1u);      // (step B: a takeover of another face's slot / a new face in an empty one)
#endif
        v = vec4(0.0); m = vec4(0.0); Pv = PL = Pl = vec2(0.0); Incon = 1.0;
    }
    m.w = Pv.x;
    // (static settle v2, XLION_VOX_FREEZE: a frozen face is read-only - no ray of anyone's writes it (a coarser level's store from a finer face's ray,
    // a second bounce's hit), not only its own pick: it stays exactly as it is until a light change (m_First bit 0) wakes every face. Before, the
    // pick alone was frozen and the other writes kept moving it - Tomas 2026-11-13: once the faces near you are frozen, no more changes)
    // (its write time still refreshed: a frozen face stays resident - unwritten for 120 frames its slot was taken over by another key and it flashed back from
    // nothing; the read-only freeze's first run flickered 0.45 for that)
    const uint Fw = VoxFreezeWord(h.x);
#ifdef GI_VOX_ONLY
    if (ConnU[2] == vox_drift_mark_v) atomicAdd(ConnU[bFreeze && !bTaken && (GIRefAccumW[Fw] & vox_frozen_v) != 0u ? 16 : !bFreeze ? 18 : g_VoxParentStore ? 19 : 17], 1u);      // (measurement: stores refused frozen / counted / freeze off (lights moving, a skinned record) / parent)
#endif
    if (bFreeze && !bTaken && (GIRefAccumW[Fw] & vox_frozen_v) != 0u) { GIRefAccumU[VoxX2Of(e)].w = floatBitsToUint(Now); return; }
    if (bTaken) GIRefAccumW[Fw] = 0u;      // (a new owner: its count starts from nothing)
    if (bFreeze && g_VoxStoreKind != 2u) VoxStoreDir(L, Org, g, N, Sample);      // (XLION_VOX_FREEZE: after the test - a frozen face's bins stay too)
    // (fireflies, kajiya's / WickedEngine's MultiscaleMeanEstimator: a sample past the face's short mean + 8 sigma + 0.1 is cut there - one bright ray (the sun,
    // a sky gap) jolted every face it landed on: the GI flickered everywhere while rays ran, Tomas 2026-11-12)
    if (v.w >= 16.0)
    {
        const float Lt = dot(m.rgb, vec3(0.2126, 0.7152, 0.0722)) + 8.0 * sqrt(max(m.w, 0.0)) + 0.1, Ls0 = dot(Sample, vec3(0.2126, 0.7152, 0.0722));
        if (Ls0 > Lt) Sample *= Lt / Ls0;
    }
    const float Ls = dot(Sample, vec3(0.2126, 0.7152, 0.0722));
    const int   ua = (a + 1) % 3, va = (a + 2) % 3;
    const vec2  Ln = vec2(N[ua], N[va]);
    if (v.w <= 0.0) { m = vec4(Sample, 0.0); Pv.y = dot(Ln, Ln); PL = Ls * Ln; Pl = Ln; }
    const float Ds = Ls - dot(m.rgb, vec3(0.2126, 0.7152, 0.0722));
#ifdef GI_VOX_ONLY
    // (against the LONG mean, 2026-11-13: the rays pick faces by their short and long means' disagreement (XLION_VOX_INCON), so the sample's
    // deviation from the short mean is biased by the pick - Police read R +.01 for ever; the long mean, once it holds 64 samples, is not)
    if (ConnU[2] == vox_drift_mark_v && g_DriftProbe && v.w >= 64.0) { const float Lk = dot(v.rgb, vec3(0.2126, 0.7152, 0.0722)) / v.w; VoxDriftAdd(Ls - Lk, Lk); }
#endif
    m.rgb += (Sample - m.rgb) * vox_short_v;
    m.w   += (Ds * Ds - m.w) * vox_short_v;
    if (v.w >= 64.0)       // (always, not only after a light change (m_First bit 1): the bounces filling in drift a face as a change does - its long mean lagged them: at 20 s 31.1 -> 28.9 % off, signed -2.68 -> -1.75, 2026-11-12)
    {
        const float Lk = dot(v.rgb, vec3(0.2126, 0.7152, 0.0722)) / v.w, Lm = dot(m.rgb, vec3(0.2126, 0.7152, 0.0722));
        const float Noise = sqrt(max(m.w, 0.0) * vox_short_v / (2.0 - vox_short_v));       // (the short mean's own standard deviation)
        Incon += (min(abs(Lm - Lk) / max(sqrt(max(m.w, 0.0)), 1e-4 + 1e-3 * Lk), 4.0) - Incon) * 0.08;
        // (2026-11-13: only while the lights moved (m_First bit 0, 0.5 s after a change) - on a settled view a face whose samples come from two differently lit
        // surfaces in its cell (a panel's recess and frame) had its short mean swing, re-anchored again and again, and popped: the blotches at Tomas's view were
        // full-trust faces. Propagation after the lights stop moves the long mean instead)
        if ((Push.m_First & 1) != 0 && abs(Lm - Lk) > vox_sigmas_v * Noise && abs(Lm - Lk) > 0.1 * Lk)
        {
#ifdef GI_VOX_ONLY
            if (ConnU[1] == vox_glide_mark_v) v *= max(0.5, 32.0 / v.w);                   // (XLION_VOX_GLIDE: its history halved, its mean kept - new samples move it faster, it glides: no jump to the short mean's noise)
            else
#endif
            v = vec4(m.rgb * 32.0, 32.0);                                                  // (it changed: the short mean is all it knows)
        }
    }
    v += vec4(Sample, 1.0);
    const float Keep = float(abs(Push.m_Epoch) & 0xFFFF);      // (the high bits: XLION_VOX_CAP)
    if (v.w >= Keep) v *= 0.5 * Keep / v.w;       // (ponytail: another lane's write to this face in this frame can be lost: samples, not light)
    const float Al = max(1.0 / v.w, 1.0 / 256.0);
    Pv.y += (dot(Ln, Ln) - Pv.y) * Al; PL += (clamp(Ls, 0.0, 60000.0) * Ln - PL) * Al; Pl += (Ln - Pl) * Al;
    // (C31: a taken slot publishes payload first, then its tag - a reader that sees the tag sees a whole record)
    GIRefAccum[e] = v;
    if (v.w < 64.0) Incon = 1.0;
    GIRefAccumU[VoxX2Of(e)] = uvec4(packHalf2x16(clamp(m.rg, vec2(0.0), vec2(60000.0))), packHalf2x16(vec2(clamp(m.b, 0.0, 60000.0), Incon)), floatBitsToUint(Now), floatBitsToUint(Now));
    if (bTaken) memoryBarrierBuffer();
    GIRefAccumU[e + 1u] = uvec4(floatBitsToUint(Tag), packHalf2x16(PL) | 1u, packHalf2x16(Pl), packHalf2x16(vec2(min(m.w, 60000.0), Pv.y)));
    // (counted and frozen only once the store is quiet - m_Frame bit 30, gpu_trace::VoxQuiet: enough rays since the last change of the lights or the eye. Before,
    // faces froze 2-5 s after the camera arrived, while the bounces were still filling in: neighbours frozen at different stages, a patchwork - 2026-11-14)
    if (bFreeze && (uint(Push.m_Frame) & 0x40000000u) != 0u && (!g_VoxParentStore || VoxFreezeParents()))
    {
        const float n = float((atomicAdd(GIRefAccumW[Fw], 1u) & ~vox_frozen_v) + 1u), Budget = VoxFreezeBudget();
        if (n >= Budget) atomicOr(GIRefAccumW[Fw], vox_frozen_v);
    }
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
    const uvec2  h = VoxHash(L, ivec3(floor(gv + 0.5)), 36u + uint(2 * a + (N[a] < 0.0 ? 1 : 0))); const uint e = VoxEntryOf(h);
    const float Tag = VoxTagOf(h), Now = float(uint(Push.m_Frame) & 0xFFFFFFu);
    const vec2  d = vec2(GIRefAccum[e + 1u].x, GIRefAccum[VoxX2Of(e)].w);
    const bool  bMine = d.x == Tag;
    vec4 c = bMine ? GIRefAccum[e] : vec4(0.0);
    if ((Push.m_First & 1) == 0 && c.w >= 1.0 && ((Push.m_First & 16) != 0 || Rnd() >= 0.125)) return c.rgb / c.w;      // (m_First's bit 4, XLION_VOX_HIT_CACHED, measurement: a hit with a kept direct never casts its shadow rays again: the best case of hits reading a cache)
    const vec3 E = VoxDirect(P, N);
    if (!bMine && d.x != 0.0 && VoxSlotInUse(e, Now)) return E;
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
    const vec3 Own = GD[15] > 0u && g_Ablate != 6 && !EmitterDirectExternal() ? VoxEmitters(P, N) : vec3(0.0);      // (XLION_VOX_ABLATE=6, measurement C38: no emitter light sampling - an emitter counts only where a ray hits it)
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
            const uvec2 h  = VoxHash(Ls, Os + ivec3(floor(gs + 0.5)), 42u + uint(2 * a + (Nh[a] < 0.0 ? 1 : 0))); const uint e = VoxEntryOf(h);
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
    if (GD[15] > 0u && g_Ablate != 6) { In += VoxEmitters(Ph, Nh); Emission = vec3(0.0); }
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
        const uvec2 h  = VoxHash(Ls, Os + ivec3(floor(gs + 0.5)), 42u + uint(2 * a + (Nk[a] < 0.0 ? 1 : 0))); const uint e = VoxEntryOf(h);
        const float Tag = VoxTagOf(h), Now = float(uint(Push.m_Frame) & 0xFFFFFFu);
        const vec2 d = vec2(GIRefAccum[e + 1u].x, GIRefAccum[VoxX2Of(e)].w);
        vec4 c = d.x == Tag ? GIRefAccum[e] : vec4(0.0);
        if (d.x != Tag && d.x != 0.0 && VoxSlotInUse(e, Now)) return;
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
    if (Push.m_Bounces == -8 || Push.m_Bounces == -9)      // (Milestone A, the movers' probes: -8 a lane a ray - Push.m_Epoch of them a probe, uniform-sphere, stratified in z - its
    {                                                       // radiance's SH to a scratch entry; -9 a lane a probe: its rays' sum into its SH by an exponential mean. 8 rays in one lane cost 1.3 ms)
        const int  K = max(Push.m_Epoch & 0xFFFF, 1);      // (bit 16: C52 narrowing - the hits unlit, no store reads or writes from a probe's ray)
        const uint i = gl_GlobalInvocationID.x, q = Push.m_Bounces == -8 ? i / uint(K) : i, m = q / 12u, p = q % 12u, Base = uint(Push.m_First);
        const vec4 H0 = Mover[Base], A = Mover[Base + 1u + 2u * m], B = Mover[Base + 2u + 2u * m];
        if (float(m) >= H0.x || A.w <= 0.0) return;      // (a block nobody holds this frame)
        const uint Scratch = vox_probe_base_v + 4u * 12u * vox_movers_v + 4u * (q * uint(K));
        if (Push.m_Bounces == -8)
        {
            const uint k = i % uint(K);
            const vec3 Pp = A.xyz + vec3(float(p & 1u) * A.w, float((p >> 1u) % 3u) * B.x, float(p / 6u) * A.w);
            const float u = (float(k) + Rnd()) / float(K), v = Rnd(), z = 1.0 - 2.0 * u, r = sqrt(max(0.0, 1.0 - z * z)), Phi = 6.2831853 * v;
            const vec3 D = vec3(r * cos(Phi), z, r * sin(Phi));
            gi_hit H; H.m_Inst = -1;
            if ((Push.m_Epoch & 0x20000) == 0) H = GICast(Pp, D, 1e30);      // (bit 17, C52 narrowing: no cast - the sky every ray)
            vec3 Lr = vec3(0.0);
            float Back = 0.0;
            if (H.m_Inst < 0) Lr = SkyAt(D);
            else
            {
                const uint HR = InstRec(H.m_Inst), HM = MeshOf(HR);
                const uvec3 HT = MTri(HM, H.m_Tri);
                const float B0 = 1.0 - H.m_U - H.m_V;
                const vec3 Nh = NormalToWorld(HR, MNrm(HM, HT.x) * B0 + MNrm(HM, HT.y) * H.m_U + MNrm(HM, HT.z) * H.m_V);
                if (dot(Nh, D) > 0.0) Back = 1.0;      // (inside something: no light from there, and a vote against the probe)
                else if ((Push.m_Epoch & 0x10000) != 0) Lr = vec3(0.0);
                else
                {
                    vec3 Albedo, Emission;
                    SurfaceAt(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Albedo, Emission);
                    const vec3 Ph = ToWorld(HR, MPos(HM, HT.x) * B0 + MPos(HM, HT.y) * H.m_U + MPos(HM, HT.z) * H.m_V);
                    vec3 In = VoxDirectKept(Ph, Nh, Hd) + VoxRead(Ph, Nh, Hd.xyz, Hd.w);      // (the voxel rays' own hit lighting: VoxRay)
                    if (GD[15] > 0u) { In += VoxEmitters(Ph, Nh); Emission = vec3(0.0); }
                    Lr = Albedo * In * (1.0 / gi_pi_v) + Emission;
                }
            }
            if (any(isnan(Lr)) || any(greaterThan(Lr, vec3(1e5)))) Lr = vec3(0.0);
            vec4 R = vec4(0.0), G = vec4(0.0), Bl = vec4(0.0);
            ShAddSample(R, G, Bl, Lr, D, float(K));
            const uint e = Scratch + 4u * k;
            GIRefAccum[e] = R; GIRefAccum[e + 1u] = G; GIRefAccum[e + 2u] = Bl; GIRefAccum[e + 3u] = vec4(Back, 0.0, 0.0, 0.0);
            return;
        }
        vec4 R = vec4(0.0), G = vec4(0.0), Bl = vec4(0.0);
        float nBack = 0.0;
        for (int k = 0; k < K; ++k) { const uint e = Scratch + 4u * uint(k); R += GIRefAccum[e]; G += GIRefAccum[e + 1u]; Bl += GIRefAccum[e + 2u]; nBack += GIRefAccum[e + 3u].x; }
        const uint e = vox_probe_base_v + 4u * (12u * m + p);
        const vec4 V = GIRefAccum[e + 3u];
        const bool bReset = B.y > 0.5 || V.y <= 0.0;
        const float a = bReset ? 1.0 : (B.z > 0.0 ? B.z : H0.y);      // (B.z: this mover's own blend - by its speed, plan_forward)
        // (validity: inside geometry - over a quarter of its rays on back faces - invalid at once; valid again only by the step a frame, ~0.5 s)
        // (2026-11-13: by the back-face share's own running mean (V.w), not this frame's 8 rays - 2 or 3 back hits of 8 by chance flipped a probe near the ground
        // invalid and back every few frames, its weight came and went in the trilinear read: the characters flickered on 18 % of their pixels)
        const float BackNow = nBack / float(K), Bf = bReset ? BackNow : mix(V.w, BackNow, a);
        const float Valid = Bf > 0.25 ? 0.0 : (bReset ? 1.0 : min(V.x + H0.z, 1.0));
        GIRefAccum[e]      = mix(GIRefAccum[e], R, a);
        GIRefAccum[e + 1u] = mix(GIRefAccum[e + 1u], G, a);
        GIRefAccum[e + 2u] = mix(GIRefAccum[e + 2u], Bl, a);
        // (z: seconds since it was last valid - the read's bounded history, v3.1 #1: an invalid probe's last estimate bridges at most 0.5 s; after a reset none)
        const float Since = Valid > 0.0 ? 0.0 : (bReset ? 1e3 : V.z + 0.5 * H0.z);
        GIRefAccum[e + 3u] = vec4(Valid, min(V.y + 1.0, 65000.0), min(Since, 1e3), Bf);
        return;
    }
#endif
    // Of a few pixels, the one whose cell knows least: the rays go where they are needed - the cells a moving eye just brought in - and not, as the screen's area would have it, to the near floor
    vec3 P = vec3(0.0), N = vec3(0.0, 1.0, 0.0);
    int L = 0; ivec3 Org = ivec3(0); vec3 g = vec3(0.0);
    float Least = 1e30, OriginAO = 1.0, LeastCount = 0.0, LeastIncon = 1.0;
    uint  OriginKind = 0u;      // (C40: 2 a skinned surface's point - its samples to the records of its kind, + 128; the gather's w 2)
    const bool bCoh = (Push.m_First & 8) != 0;
    if (bCoh)
    {
        g_CohOn = true;
        const uint Run = RHash(gl_WorkGroupID.x * 2654435761u + uint(Push.m_Frame) * 97u);      // (a group of 64 shares a direction: more shared lost directions: 21.7 against 17.9 at roughness 0.3)
        g_CohU = float(Run & 0xFFFFu) / 65536.0; g_CohV = float(Run >> 16u) / 65536.0;
    }
    const float VoxIncon = float((abs(Push.m_Epoch) >> 16) & 0x7FFF) / 100.0;      // (XLION_VOX_INCON x 100 in m_Epoch's bits 16-30; 0: the least-known pick)
    // (Y4, 2026-11-13: the picks choose faces by their state - the most inconsistent, or the least known - so their samples drift by construction and
    // the store-wide drift never settled (Police R +.01 for ever). One pick in 16 is a probe instead: a random pixel, taken as it comes)
#ifdef GI_VOX_ONLY
    g_DriftProbe = ConnU[2] == vox_drift_mark_v && !bCoh && Rnd() < 0.0625;
#endif
    for (int k = 0; k < (bCoh || g_DriftProbe ? 1 : VOX_PICKS); ++k)
    {
        const uint  px = bCoh ? (RHash(uint(Push.m_Frame)) + gl_GlobalInvocationID.x * 7u) % uint(Push.m_nPixels) : uint(Rnd() * float(Push.m_nPixels)) % uint(Push.m_nPixels);
        const vec4  Ps = GIRefSample[2u * px];
        if (Ps.w < 0.5) continue;
        const vec3  Nk = GIRefSample[2u * px + 1u].xyz;
        int Lk; ivec3 Ok; vec3 gk;
        if (!VoxCell(Ps.xyz, Hd.xyz, Hd.w, 0, Lk, Ok, gk)) continue;
        const int   Ax = abs(Nk.x) >= abs(Nk.y) && abs(Nk.x) >= abs(Nk.z) ? 0 : abs(Nk.y) >= abs(Nk.z) ? 1 : 2;
        vec3 gq = vec3(Ok) + gk; gq[Ax] *= vox_layers_v;
        const uint  Kk = Ps.w > 1.5 ? 2u : 0u;
        const uvec2  hk = VoxHash(Lk, ivec3(floor(gq + 0.5)), uint(2 * Ax + (Nk[Ax] < 0.0 ? 1 : 0)) + 64u * Kk); const uint ek = VoxEntryOf(hk);
        const bool bOwn = GIRefAccum[ek + 1u].x == VoxTagOf(hk);
        float n = bOwn ? GIRefAccum[ek].w : 0.0;
        const float nCount = n, nIncon = bOwn ? unpackHalf2x16(floatBitsToUint(GIRefAccum[VoxX2Of(ek)].y)).y : 1.0;
        {   // (settle phase 1: the faces this seen point reads, marked seen this frame. 2026-11-13: every axis face its normal weighs (the gather reads all three by
            // N x N, not only the dominant one) at this level and the next coarser (the gather falls back to it) - before, the other axes' and the parent's faces
            // were 'unused' after 2 s, taken over by off-screen hit keys, and restarted from nothing: ~350 takeovers a second on a settled still view, the blotches)
            const float NowS = float(uint(Push.m_Frame) & 0xFFFFFFu);
            const vec3 Wn = Nk * Nk;
            for (int lv = 0; lv < 2; ++lv)
            {
                int Lm; ivec3 Om; vec3 gm;
                if (lv == 0) { Lm = Lk; Om = Ok; gm = gk; } else if (!VoxCell(Ps.xyz, Hd.xyz, Hd.w, Lk + 1, Lm, Om, gm)) break;
                for (int ax = 0; ax < 3; ++ax)
                {
                    if (Wn[ax] < 0.05) continue;
                    vec3 gv = vec3(Om) + gm; gv[ax] *= vox_layers_v;
                    const ivec3 i0 = ivec3(floor(gv));
                    const uint Fc = uint(2 * ax + (Nk[ax] < 0.0 ? 1 : 0)) + 64u * Kk;
                    for (int c = 0; c < 8; ++c)
                    {
                        const uvec2 hc = VoxHash(Lm, i0 + ivec3(c & 1, (c >> 1) & 1, (c >> 2) & 1), Fc); const uint ec = VoxEntryOf(hc);
                        if (GIRefAccum[ec + 1u].x == VoxTagOf(hc) && GIRefAccum[VoxX2Of(ec)].z != NowS) { GIRefAccumU[VoxX2Of(ec)].z = floatBitsToUint(NowS);
#ifdef GI_VOX_ONLY
                            if (ConnU[2] == vox_drift_mark_v) atomicAdd(ConnU[11], 1u);
#endif
                        }
                    }
                }
            }
        }
        // (XLION_VOX_FREEZE: a frozen face isn't a candidate at all - it used to win the pick when the others were calm too and then fire nothing; the review 2026-11-13)
        if (bOwn && Kk != 2u && VoxFreezeOn() && (GIRefAccumW[VoxFreezeWord(hk.x)] & vox_frozen_v) != 0u)
        {   // (XLION_VOX_FREEZE_PARENTS: a frozen face still takes rays while its parent (the next level, which the read falls back to and blends) isn't frozen - its own stores refused, the parent's counted)
            if (!VoxFreezeParents()) continue;
            int Lp; ivec3 Op; vec3 gp;
            if (!VoxCell(Ps.xyz, Hd.xyz, Hd.w, Lk + 1, Lp, Op, gp)) continue;
            vec3 gpq = vec3(Op) + gp; gpq[Ax] *= vox_layers_v;
            const uvec2 hp = VoxHash(Lp, ivec3(floor(gpq + 0.5)), uint(2 * Ax + (Nk[Ax] < 0.0 ? 1 : 0))); const uint ep = VoxEntryOf(hp);
            if (GIRefAccum[ep + 1u].x == VoxTagOf(hp) && (GIRefAccumW[VoxFreezeWord(hp.x)] & vox_frozen_v) != 0u) continue;
        }
        if (VoxIncon > 0.0) n = n < 64.0 ? n - 1e6 : -unpackHalf2x16(floatBitsToUint(GIRefAccum[VoxX2Of(ek)].y)).y;      // (XLION_VOX_INCON: the unknown first, then the most inconsistent)
        if (n < Least) { Least = n; LeastCount = nCount; LeastIncon = nIncon; P = Ps.xyz; N = Nk; L = Lk; Org = Ok; g = gk; OriginAO = GIRefSample[2u * px + 1u].w; OriginKind = Kk; }
    }
    if (Least >= 1e30) return;
    // (XLION_VOX_INCON=<scale>: a known face takes the ray with probability inconsistency x scale - a resolved face (its long and short means agree within
    // their noise) takes none: no work, no flicker; one still drifting, propagating, stays inconsistent and keeps its rays. WickedEngine: rays = inconsistency x boost)
    // (VOX_AMORTIZE, WickedEngine's temporal ray amortization: a known face in a coarser (farther) level takes its rays 2^-level as often - far light is low
    // frequency and its faces long-lived; new faces (the unknown) are not slowed)
    // (settle phase 1, XLION_VOX_FREEZE=1 - m_First bit 16 - an experiment, off by default after the review: converged - half the window of samples or more, short and long means agreeing within 2.5 sigma of the short mean - no ray
    // at all while the lights are still: it stays as it is until a light change wakes every face. Tomas 2026-11-13: settled faces stole rays and made noise)
    if (!g_DriftProbe && VoxIncon > 0.0 && Least > -1e5 && Rnd() >= min(-Least * VoxIncon, 1.0) * exp2(-VOX_AMORTIZE * float(L))) return;
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
            g_VoxStoreKind = r == 0 && OriginKind == 2u ? 2u : r == 0 && BucketAO > 0.0 && OriginAO < BucketAO ? 1u : 0u;
            VoxStore(L, Org, g, N, Sample);
            if ((Push.m_First & 2048) != 0 && g_VoxStoreKind != 2u) VoxDirsAdd(P, N, Hd.w, g_HasSeen ? g_SeenSample : Sample);
            int Lp; ivec3 Op; vec3 gp;
            if (r == 0 && VoxCell(P, Hd.xyz, Hd.w, L + 1, Lp, Op, gp)) { g_VoxParentStore = true; VoxStore(Lp, Op, gp, N, Sample); g_VoxParentStore = false; }
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
