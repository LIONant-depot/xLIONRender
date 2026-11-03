#version 450
// The GI's ground truth for experiments (XLION_GI_REF, xLION_Docs gi_next_level_analysis.md): a brute path tracer through the GI's BVH, for
// the points the gather wrote last frame (no surfels read: the sun, lamps, sky and emission at each hit, Bounces diffuse bounces with the
// hits' albedo). Each pixel's irradiance averaged over the frames while its point stays where it is. Slow on purpose; never for a game.
// A pixel a lane; m_First: the next pixel (wraps); m_Epoch: the lights' version (another: the pixel starts over)
layout(local_size_x = 4) in;      // (a few lanes: lanes in a group pay for each other: 64 rays cost 0.9 ms in groups of 4, 2.9 in one of 64; 32768 cost 9 and 4: launching groups isn't free)

layout(std430, set = 0, binding = 5) readonly buffer samples_b { vec4 GIRefSample[]; };     // per pixel: its world point, w 1 (0: none); then its normal
layout(std430, set = 0, binding = 6)          buffer accum_b   { vec4 GIRefAccum[]; };      // per pixel 2: the sum (rgb) and count, its point
layout(push_constant) uniform push_t { int m_nPixels, m_First, m_Count, m_Frame, m_TlasRoot, m_PrimOffset, m_Bounces, m_Epoch; } Push;
#include "xlionrender_gi_pass_common.glsl"

uint RHash(uint x) { x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16; return x; }
uint g_Seed;
float Rnd() { g_Seed = RHash(g_Seed + 0x9E3779B9u); return float(g_Seed) / 4294967296.0; }
vec3 CosineDir(in const vec3 N)
{
    const float u = Rnd(), v = Rnd(), r = sqrt(u), Phi = 6.2831853 * v;
    const vec3  T = normalize(cross(abs(N.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0), N)), B = cross(N, T);
    return normalize(T * (r * cos(Phi)) + B * (r * sin(Phi)) + N * sqrt(max(0.0, 1.0 - u)));
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
void VoxStore(in const int L, in const ivec3 Org, in const vec3 g, in const vec3 N, in const vec3 Sample)
{
    const vec3  w = N * N;
    const float r = Rnd() * (w.x + w.y + w.z);
    const int   a = r < w.x ? 0 : r < w.x + w.y ? 1 : 2;
    vec3 gv = vec3(Org) + g; gv[a] *= vox_layers_v;
    const ivec3 i = ivec3(floor(gv)); const vec3 f = gv - vec3(i);
    const ivec3 q = i + ivec3(Rnd() < f.x ? 1 : 0, Rnd() < f.y ? 1 : 0, Rnd() < f.z ? 1 : 0);
    const uint  h = VoxHash(L, q, uint(2 * a + (N[a] < 0.0 ? 1 : 0))), e = VoxEntryOf(h);
    const float Tag = VoxTagOf(h), Now = float(uint(Push.m_Frame) & 0xFFFFFFu);
    vec4 v = GIRefAccum[e], m = GIRefAccum[e + 1u];
    const vec4 d = GIRefAccum[e + 2u];
    if (d.x != Tag)
    {
        // someone else's: taken only if nobody has fed it for 2 seconds or so: a face in use keeps its slot and its history (a face that finds no room waits)
        if (d.x != 0.0 && abs(Now - d.y) < 120.0) return;
        v = vec4(0.0); m = vec4(0.0);
    }
    const float Ls = dot(Sample, vec3(0.2126, 0.7152, 0.0722));
    if (v.w <= 0.0) m = vec4(Sample, 0.0);
    const float Ds = Ls - dot(m.rgb, vec3(0.2126, 0.7152, 0.0722));
    m.rgb += (Sample - m.rgb) * vox_short_v;
    m.w   += (Ds * Ds - m.w) * vox_short_v;
    if (v.w >= 64.0 && Push.m_First != 0)       // (m_First: the lights changed lately)
    {
        const float Lk = dot(v.rgb, vec3(0.2126, 0.7152, 0.0722)) / v.w, Lm = dot(m.rgb, vec3(0.2126, 0.7152, 0.0722));
        const float Noise = sqrt(max(m.w, 0.0) * vox_short_v / (2.0 - vox_short_v));       // (the short mean's own standard deviation)
        if (abs(Lm - Lk) > vox_sigmas_v * Noise && abs(Lm - Lk) > 0.1 * Lk)
            v = vec4(m.rgb * 32.0, 32.0);                                                  // (it changed: the short mean is all it knows)
    }
    v += vec4(Sample, 1.0);
    if (v.w >= float(abs(Push.m_Epoch))) v *= 0.5 * float(abs(Push.m_Epoch)) / v.w;       // (ponytail: another lane's write to this face in this frame can be lost: samples, not light)
    GIRefAccum[e] = v; GIRefAccum[e + 1u] = m; GIRefAccum[e + 2u] = vec4(Tag, Now, 0.0, 0.0);
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
// A cosine ray from a surface point: pi x what it sees (a sample of the point's indirect irradiance), and where it hit
vec3 VoxRay(in const vec3 P, in const vec3 N, in const vec4 Hd, out vec3 Ph, out vec3 Nh, out bool bHit)
{
    bHit = false; Ph = vec3(0.0); Nh = vec3(0.0, 1.0, 0.0);
    // the emitters by light sampling (G5, as the surfels' pass: the table, a shadow ray), this point's own unless lighting shades it; an emission a ray hits is not counted again
    const vec3 Own = GD[15] > 0u && !EmitterDirectExternal() ? VoxEmitters(P, N) : vec3(0.0);
    const vec3 D = CosineDir(N);
    const gi_hit H = GICast(P + N * 2e-3, D, 1e30);
    if (H.m_Inst < 0) return Own + SkyAt(D) * gi_pi_v;
    const uint HR = InstRec(H.m_Inst), HM = MeshOf(HR);
    const uvec3 HT = MTri(HM, H.m_Tri);
    const float B0 = 1.0 - H.m_U - H.m_V;
    Nh = NormalToWorld(HR, MNrm(HM, HT.x) * B0 + MNrm(HM, HT.y) * H.m_U + MNrm(HM, HT.z) * H.m_V);
    if (dot(Nh, D) > 0.0) return Own;
    vec3 Albedo, Emission;
    SurfaceAt(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Albedo, Emission);
    Ph = ToWorld(HR, MPos(HM, HT.x) * B0 + MPos(HM, HT.y) * H.m_U + MPos(HM, HT.z) * H.m_V);
    bHit = true;
    vec3 In = Direct(Ph, Nh) + VoxRead(Ph, Nh, Hd.xyz, Hd.w);
    if (GD[15] > 0u) { In += VoxEmitters(Ph, Nh); Emission = vec3(0.0); }
    return Own + (Albedo * In * (1.0 / gi_pi_v) + Emission) * gi_pi_v;
}
void VoxMain()
{
    if (Push.m_Bounces == -2) { for (uint k = 0u; k < 16u; ++k) { const uint i = gl_GlobalInvocationID.x * 16u + k; if (int(i) < Push.m_Count) GIRefAccum[uint(Push.m_First) + i] = vec4(0.0); } return; }     // (a lane clears 16: not a million groups)
    if (int(gl_GlobalInvocationID.x) >= Push.m_Count) return;
    g_GIBvhTlasRoot = Push.m_TlasRoot; g_GIBvhPrimOffset = Push.m_PrimOffset; g_GIBvhSeeThrough = true;
    const vec4 Hd = GIRefSample[2u * uint(Push.m_nPixels)];
    if (Hd.w <= 0.0) return;
    g_Seed = RHash(gl_GlobalInvocationID.x * 2654435761u + uint(Push.m_Frame) * 97u);
    // Of a few pixels, the one whose cell knows least: the rays go where they are needed - the cells a moving eye just brought in - and not, as the screen's area would have it, to the near floor
    vec3 P = vec3(0.0), N = vec3(0.0, 1.0, 0.0);
    int L = 0; ivec3 Org = ivec3(0); vec3 g = vec3(0.0);
    float Least = 1e30;
    for (int k = 0; k < 4; ++k)
    {
        const uint  px = uint(Rnd() * float(Push.m_nPixels)) % uint(Push.m_nPixels);
        const vec4  Ps = GIRefSample[2u * px];
        if (Ps.w < 0.5) continue;
        const vec3  Nk = GIRefSample[2u * px + 1u].xyz;
        int Lk; ivec3 Ok; vec3 gk;
        if (!VoxCell(Ps.xyz, Hd.xyz, Hd.w, 0, Lk, Ok, gk)) continue;
        const int   Ax = abs(Nk.x) >= abs(Nk.y) && abs(Nk.x) >= abs(Nk.z) ? 0 : abs(Nk.y) >= abs(Nk.z) ? 1 : 2;
        vec3 gq = vec3(Ok) + gk; gq[Ax] *= vox_layers_v;
        const float n  = VoxFace(Lk, ivec3(floor(gq + 0.5)), uint(2 * Ax + (Nk[Ax] < 0.0 ? 1 : 0))).w;
        if (n < Least) { Least = n; P = Ps.xyz; N = Nk; L = Lk; Org = Ok; g = gk; }
    }
    if (Least >= 1e30) return;
    vec3 Ph, Nh; bool bHit;
    const vec3 Sample = VoxRay(P, N, Hd, Ph, Nh, bHit);
    if (!any(isnan(Sample)) && all(lessThan(Sample, vec3(1e6))))
    {
        VoxStore(L, Org, g, N, Sample);
        int Lp; ivec3 Op; vec3 gp;
        if (VoxCell(P, Hd.xyz, Hd.w, L + 1, Lp, Op, gp)) VoxStore(Lp, Op, gp, N, Sample);
    }
    // the surface it hit too, seen yet or not: one more ray from there is a sample of that surface's own indirect light. Off screen surfaces are lit before the eye turns to them
    if (Push.m_Epoch < 0 && bHit && VoxCell(Ph, Hd.xyz, Hd.w, 0, L, Org, g))
    {
        vec3 Ph2, Nh2; bool bHit2;
        const vec3 S2 = VoxRay(Ph, Nh, Hd, Ph2, Nh2, bHit2);
        if (!any(isnan(S2)) && all(lessThan(S2, vec3(1e6)))) VoxStore(L, Org, g, Nh, S2);
    }
}

void main()
{
    if (Push.m_Bounces < 0) { VoxMain(); return; }
    if (int(gl_GlobalInvocationID.x) >= Push.m_Count) return;
    g_GIBvhTlasRoot = Push.m_TlasRoot; g_GIBvhPrimOffset = Push.m_PrimOffset; g_GIBvhSeeThrough = true;
    const uint i = (uint(Push.m_First) + gl_GlobalInvocationID.x) % uint(Push.m_nPixels);
    const vec4 Ps = GIRefSample[2u * i];
    if (Ps.w < 0.5) return;
    const vec3 N0 = GIRefSample[2u * i + 1u].xyz;
    g_Seed = RHash(i * 2654435761u + uint(Push.m_Frame) * 97u);
    vec3 P = Ps.xyz, N = N0, Th = vec3(1.0), Sum = vec3(0.0);
    if (GD[15] > 0u && !EmitterDirectExternal()) Sum += EmitterIrradianceAt(P, N, g_Seed) * (1.0 / gi_pi_v);      // (the point's own: lighting doesn't shade them, the GI shows them)
    for (int b = 0; b <= Push.m_Bounces; ++b)
    {
        const vec3 D = CosineDir(N);
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
    if (all(lessThan(Sum, vec3(1e6))) && !any(isnan(Sum))) A += vec4(Sum * gi_pi_v, 1.0);
    GIRefAccum[2u * i] = A;
}
