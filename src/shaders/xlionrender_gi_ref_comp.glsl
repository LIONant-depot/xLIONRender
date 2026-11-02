#version 450
// The GI's ground truth for experiments (XLION_GI_REF, xLION_Docs gi_next_level_analysis.md): a brute path tracer through the GI's BVH, for
// the points the gather wrote last frame (no surfels read: the sun, lamps, sky and emission at each hit, Bounces diffuse bounces with the
// hits' albedo). Each pixel's irradiance averaged over the frames while its point stays where it is. Slow on purpose; never for a game.
// A pixel a lane; m_First: the next pixel (wraps); m_Epoch: the lights' version (another: the pixel starts over)
layout(local_size_x = 64) in;

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
// around it, a face its normal looks toward, both picked by their read weights. m_Bounces: -1 this, -2 clear; m_Epoch: the samples a face keeps.
// The header after the pixels (2 entries): the anchor and level 0's cell, the eye
const int vox_dim_v = 64, vox_levels_v = 4;
bool VoxCell(in const vec3 Pw, in const vec3 Anchor, in const float Cell0, out int L, out ivec3 Org, out vec3 g)
{
    for (L = 0; L < vox_levels_v; ++L)
    {
        const float Cell = Cell0 * float(1 << L);
        Org = ivec3(floor(Anchor / Cell)) - ivec3(vox_dim_v / 2);
        g   = Pw / Cell - vec3(Org) - 0.5;
        if (all(greaterThanEqual(g, vec3(0.0))) && all(lessThan(g, vec3(vox_dim_v - 1)))) return true;
    }
    return false;
}
uint VoxEntry(in const int L, in const ivec3 q, in const uint Face) { return (uint(L) * uint(vox_dim_v * vox_dim_v * vox_dim_v) + uint(q.x) + uint(q.y) * uint(vox_dim_v) + uint(q.z) * uint(vox_dim_v * vox_dim_v)) * 6u + Face; }
vec3 VoxRead(in const vec3 Pw, in const vec3 n, in const vec3 Anchor, in const float Cell0)
{
    int L; ivec3 Org; vec3 g;
    if (!VoxCell(Pw, Anchor, Cell0, L, Org, g)) return vec3(0.0);
    const ivec3 i = ivec3(floor(g)); const vec3 f = g - vec3(i);
    vec3 Sum = vec3(0.0); float SumW = 0.0;
    for (int a = 0; a < 3; ++a)
    {
        const float wa = n[a] * n[a];
        if (wa < 0.02) continue;
        vec3 Num = vec3(0.0); float Den = 0.0;
        for (int c = 0; c < 8; ++c)
        {
            const ivec3 o = ivec3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
            const float t = (o.x != 0 ? f.x : 1.0 - f.x) * (o.y != 0 ? f.y : 1.0 - f.y) * (o.z != 0 ? f.z : 1.0 - f.z);
            const vec4  v = GIRefAccum[VoxEntry(L, i + o, uint(2 * a + (n[a] < 0.0 ? 1 : 0)))];
            Num += v.rgb * t; Den += v.w * t;
        }
        if (Den > 1e-3) { Sum += Num / Den * wa; SumW += wa; }
    }
    return SumW > 0.0 ? Sum / SumW : vec3(0.0);
}
void VoxMain()
{
    if (Push.m_Bounces == -2) { if (int(gl_GlobalInvocationID.x) < Push.m_Count) GIRefAccum[uint(Push.m_First) + gl_GlobalInvocationID.x] = vec4(0.0); return; }
    if (int(gl_GlobalInvocationID.x) >= Push.m_Count) return;
    g_GIBvhTlasRoot = Push.m_TlasRoot; g_GIBvhPrimOffset = Push.m_PrimOffset; g_GIBvhSeeThrough = true;
    const vec4 Hd = GIRefSample[2u * uint(Push.m_nPixels)];
    if (Hd.w <= 0.0) return;
    g_Seed = RHash(gl_GlobalInvocationID.x * 2654435761u + uint(Push.m_Frame) * 97u);
    const uint px = uint(Rnd() * float(Push.m_nPixels)) % uint(Push.m_nPixels);
    const vec4 Ps = GIRefSample[2u * px];
    if (Ps.w < 0.5) return;
    const vec3 P = Ps.xyz, N = GIRefSample[2u * px + 1u].xyz;
    int L; ivec3 Org; vec3 g;
    if (!VoxCell(P, Hd.xyz, Hd.w, L, Org, g)) return;
    const vec3 D = CosineDir(N);
    const gi_hit H = GICast(P + N * 2e-3, D, 1e30);
    vec3 Sample = vec3(0.0);
    if (H.m_Inst < 0) Sample = SkyAt(D);
    else
    {
        const uint HR = InstRec(H.m_Inst), HM = MeshOf(HR);
        const uvec3 HT = MTri(HM, H.m_Tri);
        const float B0 = 1.0 - H.m_U - H.m_V;
        const vec3 Nh = NormalToWorld(HR, MNrm(HM, HT.x) * B0 + MNrm(HM, HT.y) * H.m_U + MNrm(HM, HT.z) * H.m_V);
        if (dot(Nh, D) <= 0.0)
        {
            vec3 Albedo, Emission;
            SurfaceAt(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Albedo, Emission);
            const vec3 Ph = ToWorld(HR, MPos(HM, HT.x) * B0 + MPos(HM, HT.y) * H.m_U + MPos(HM, HT.z) * H.m_V);
            vec3 In = Direct(Ph, Nh) + VoxRead(Ph, Nh, Hd.xyz, Hd.w);
            if (EmitterDirectExternal()) { In += EmitterIrradianceAt(Ph, Nh, g_Seed); Emission = vec3(0.0); }
            Sample = Albedo * In * (1.0 / gi_pi_v) + Emission;
        }
    }
    if (any(isnan(Sample)) || !all(lessThan(Sample, vec3(1e6)))) return;
    // the cell and the face, by the read's weights: the samples land where the reads look for them
    const ivec3 i = ivec3(floor(g)); const vec3 f = g - vec3(i);
    const ivec3 q = i + ivec3(Rnd() < f.x ? 1 : 0, Rnd() < f.y ? 1 : 0, Rnd() < f.z ? 1 : 0);
    const vec3  w = N * N;
    const float r = Rnd() * (w.x + w.y + w.z);
    const int   a = r < w.x ? 0 : r < w.x + w.y ? 1 : 2;
    const uint  e = VoxEntry(L, q, uint(2 * a + (N[a] < 0.0 ? 1 : 0)));
    vec4 v = GIRefAccum[e] + vec4(Sample * gi_pi_v, 1.0);
    if (v.w >= float(Push.m_Epoch)) v *= 0.5;       // (ponytail: another lane's write to this face in this frame can be lost: samples, not light)
    GIRefAccum[e] = v;
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
        if (EmitterDirectExternal()) { In += EmitterIrradianceAt(Ph, Nh, g_Seed); Emission = vec3(0.0); }
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
