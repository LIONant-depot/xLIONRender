#version 450
// The GI's probe grid (gpu_trace::UpdateProbes): irradiance for what has no surfels of its own - skinned and moving meshes. A grid of
// probes around the camera (toroidal: a probe's slot is its world cell modulo the grid; a moved grid keeps the probes still in it), each
// an L1 spherical harmonic of the radiance its rays see: at a hit the light the GI pass gives a hit (the direct light, the surfels'
// read, the albedo, the emission), past the scene the sky. One workgroup a probe, a ray a lane, a slice of the probes a frame; blended
// into the last (a probe that moved to another cell starts over). A probe whose rays mostly meet back faces is inside a wall: flagged,
// the reads skip it (xlionrender_surfel_gi.glsl SurfelGIProbes)
// The probe buffer (words): a header (16) - its first cell (ivec3), the spacing, the grid's size (ivec3), the probes' count; then per
// probe 16: the SH (4 coefficients x RGB, floats), its world cell (ivec3), its flags (bit 0: lit, bit 1: inside)
layout(local_size_x = 64) in;

layout(std430, set = 0, binding = 5) buffer probes_b { uint GP[]; };
layout(push_constant) uniform push_t { int m_nProbes, m_TlasRoot, m_PrimOffset, m_First, m_Frame, m_Count, m_Pad0, m_Pad1; } Push;
#include "xlionrender_gi_pass_common.glsl"

const uint  gi_probe_words_v = 16u, gi_probe_header_v = 16u;
const float gi_sh0_v = 0.282095, gi_sh1_v = 0.488603;
shared vec3 s_SH[64][4];
shared uint s_Back;

ivec3 PMod(ivec3 a, ivec3 m) { return a - m * ivec3(floor(vec3(a) / vec3(m))); }     // (GLSL's % of a negative is undefined)
uint PHash(uint x) { x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16; return x; }

void main()
{
    g_GIBvhTlasRoot = Push.m_TlasRoot; g_GIBvhPrimOffset = Push.m_PrimOffset; g_GIBvhSeeThrough = true;
    const uint lid = gl_LocalInvocationID.x;
    const uint p = uint(Push.m_First + int(gl_WorkGroupID.x)) % uint(max(Push.m_nProbes, 1));
    const ivec3 Org = ivec3(int(GP[0]), int(GP[1]), int(GP[2])), Dim = ivec3(int(GP[4]), int(GP[5]), int(GP[6]));
    const float Sp = uintBitsToFloat(GP[3]);
    // the slot's cell in the grid now: the one of the grid's cells that lands on this slot
    const ivec3 Lc = ivec3(int(p) % Dim.x, (int(p) / Dim.x) % Dim.y, int(p) / (Dim.x * Dim.y));
    const ivec3 Cell = Org + PMod(Lc - Org, Dim);
    const vec3  Pp = vec3(Cell) * Sp;
    if (lid == 0u) s_Back = 0u;
    barrier();
    // a ray a lane: the spherical Fibonacci set, turned by the frame (the frames' rays fill the sphere in)
    const float i = float(lid) + 0.5, Phi = 2.39996323 * i + float(PHash(uint(Push.m_Frame) * 977u + p) & 0xFFFFu) * (6.2831853 / 65536.0);
    const float Ct = 1.0 - 2.0 * i / 64.0, St = sqrt(max(0.0, 1.0 - Ct * Ct));
    const vec3  D = vec3(cos(Phi) * St, Ct, sin(Phi) * St);
    vec3 L = vec3(0);
    const gi_hit H = GICast(Pp, D, 1e30);
    if (H.m_Inst < 0) L = SkyAt(D);
    else
    {
        const uint HR = InstRec(H.m_Inst), HM = MeshOf(HR);
        const uvec3 HT = MTri(HM, H.m_Tri);
        const float B0 = 1.0 - H.m_U - H.m_V;
        const vec3 Nh = NormalToWorld(HR, MNrm(HM, HT.x) * B0 + MNrm(HM, HT.y) * H.m_U + MNrm(HM, HT.z) * H.m_V);
        if (dot(Nh, D) > 0.0) atomicAdd(s_Back, 1u);
        else
        {
            vec3 Albedo, Emission, Q;
            SurfaceAt(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Albedo, Emission);
            const vec3 Ph = ToWorld(HR, MPos(HM, HT.x) * B0 + MPos(HM, HT.y) * H.m_U + MPos(HM, HT.z) * H.m_V);
            const vec3 In = Direct(Ph, Nh) + (Indirect(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Q) ? Q : Unsupported(H.m_Inst, H.m_Tri, H.m_U, H.m_V));
            L = Albedo * In * (1.0 / gi_pi_v) + Emission;
        }
    }
    // its share of the L1 projection (4 pi / 64 rays)
    const float w = 4.0 * gi_pi_v / 64.0;
    s_SH[lid][0] = L * (gi_sh0_v * w); s_SH[lid][1] = L * (gi_sh1_v * D.x * w); s_SH[lid][2] = L * (gi_sh1_v * D.y * w); s_SH[lid][3] = L * (gi_sh1_v * D.z * w);
    barrier();
    for (uint s = 32u; s > 0u; s >>= 1)
    {
        if (lid < s) for (int k = 0; k < 4; ++k) s_SH[lid][k] += s_SH[lid + s][k];
        barrier();
    }
    if (lid != 0u) return;
    const uint B = gi_probe_header_v + p * gi_probe_words_v;
    const bool bSame = (GP[B + 15u] & 1u) != 0u && ivec3(int(GP[B + 12u]), int(GP[B + 13u]), int(GP[B + 14u])) == Cell;
    const float a = bSame ? 0.2 : 1.0;                 // (a probe of another cell starts over)
    for (int k = 0; k < 4; ++k)
    {
        const vec3 Old = vec3(uintBitsToFloat(GP[B + 3u * uint(k)]), uintBitsToFloat(GP[B + 3u * uint(k) + 1u]), uintBitsToFloat(GP[B + 3u * uint(k) + 2u]));
        const vec3 New = mix(Old, s_SH[0][k], a);
        GP[B + 3u * uint(k)] = floatBitsToUint(New.x); GP[B + 3u * uint(k) + 1u] = floatBitsToUint(New.y); GP[B + 3u * uint(k) + 2u] = floatBitsToUint(New.z);
    }
    GP[B + 12u] = uint(Cell.x); GP[B + 13u] = uint(Cell.y); GP[B + 14u] = uint(Cell.z);
    GP[B + 15u] = 1u | (s_Back > 16u ? 2u : 0u);      // (a quarter of its rays on back faces: inside)
}
