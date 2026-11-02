#version 450
// The GI's probe grid (gpu_trace::UpdateProbes): irradiance for what has no surfels of its own - skinned and moving meshes. A grid of
// probes around the camera (toroidal: a probe's slot is its world cell modulo the grid; a moved grid keeps the probes still in it), each
// an L1 spherical harmonic of the radiance its rays see: at a hit the light the GI pass gives a hit (the direct light, the surfels'
// read, the albedo, the emission), past the scene the sky. One workgroup a probe, a ray a lane, a slice of the probes a frame; blended
// into the last (a probe that moved to another cell starts over). A probe whose rays mostly meet back faces is inside a wall: flagged,
// the reads skip it (xlionrender_surfel_gi.glsl SurfelGIProbes)
// The probe buffer (words): a header (16) - its first cell (ivec3), the spacing, the grid's size (ivec3), the probes' count; then per
// probe 16: the SH (4 coefficients x RGB, floats), its world cell (ivec3), its flags (bit 0: lit, bit 1: inside, bits 8-15: the share of its rays on back faces, blended: the reads' weight - a yes or no flipped with each update's random rays, whole patches jumped). Cascades: more such grids after
// it (header word 11: the next one's first word, 0: the last); m_Base: this dispatch's
layout(local_size_x = 64) in;

layout(std430, set = 0, binding = 5) buffer probes_b { uint GP[]; };
layout(std430, set = 0, binding = 6) readonly buffer list_b { uint GList[]; };     // this frame's slots, a workgroup each (the readers' near the eye first, then a round robin)
layout(push_constant) uniform push_t { int m_nProbes, m_TlasRoot, m_PrimOffset, m_First, m_Frame, m_Count, m_Base, m_HitsReadProbes; } Push;
#include "xlionrender_gi_pass_common.glsl"

const uint  gi_probe_words_v = 16u, gi_probe_header_v = 16u;
const float gi_sh0_v = 0.282095, gi_sh1_v = 0.488603;
shared vec3 s_SH[64][4];
shared uint s_Back;

ivec3 PMod(ivec3 a, ivec3 m) { return a - m * ivec3(floor(vec3(a) / vec3(m))); }     // (GLSL's % of a negative is undefined)
uint PHash(uint x) { x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16; return x; }

// The irradiance of L1 radiance toward N, each channel (Hazel, Geomerics 2011, "Reconstructing diffuse lighting from SH data"): the linear read,
// max(0, L0 + L1 . N), goes to 0 on the side away from a strong light - a shaded wall facing away from a sunlit one read black. Never negative, the same mean
float ProbeIrradiance1(in const float c0, in const vec3 c1, in const vec3 N)
{
    const float R0 = 3.14159265 * 0.282095 * c0;
    if (R0 <= 0.0) return 0.0;
    const vec3  L = 2.0943951 * 0.488603 * c1;
    const float Ll = length(L);
    if (Ll <= 1e-8 * R0) return R0;
    const float R = min(0.5 * Ll / R0, 0.999), q = 0.5 * (1.0 + dot(L / Ll, N)), p = 1.0 + 2.0 * R, a = (1.0 - R) / (1.0 + R);
    return R0 * (a + (1.0 - a) * (p + 1.0) * pow(max(q, 0.0), p));
}
vec3 ProbeIrradiance(in const vec3 c0, in const vec3 c1, in const vec3 c2, in const vec3 c3, in const vec3 N)
{
    return vec3(ProbeIrradiance1(c0.x, vec3(c1.x, c2.x, c3.x), N), ProbeIrradiance1(c0.y, vec3(c1.y, c2.y, c3.y), N), ProbeIrradiance1(c0.z, vec3(c1.z, c2.z, c3.z), N));
}
// What a hit reads of the grids (m_HitsReadProbes): the finest with a lit probe around it (SurfelGIProbes' read, without its fades)
vec3 ProbesAt(in const vec3 P, in const vec3 N)
{
    uint G0 = 0u;
    for (int c = 0; c < 4; ++c)
    {
        const ivec3 Org = ivec3(int(GP[G0]), int(GP[G0 + 1u]), int(GP[G0 + 2u])), Dim = ivec3(int(GP[G0 + 4u]), int(GP[G0 + 5u]), int(GP[G0 + 6u]));
        const float Sp = uintBitsToFloat(GP[G0 + 3u]);
        const vec3  F = P / Sp, B = floor(F), Fr = F - B;
        vec3 Sum = vec3(0.0); float W = 0.0;
        for (int k = 0; k < 8; ++k)
        {
            const ivec3 O = ivec3(k & 1, (k >> 1) & 1, (k >> 2) & 1), C = ivec3(B) + O;
            if (any(lessThan(C, Org)) || any(greaterThanEqual(C, Org + Dim))) continue;
            const ivec3 Lc = PMod(C, Dim);
            const uint  b = G0 + gi_probe_header_v + gi_probe_words_v * uint(Lc.x + Dim.x * (Lc.y + Dim.y * Lc.z));
            if ((GP[b + 15u] & 1u) == 0u || ivec3(int(GP[b + 12u]), int(GP[b + 13u]), int(GP[b + 14u])) != C) continue;
            const vec3  T = mix(vec3(1.0) - Fr, Fr, vec3(O));
            const vec3  Dp = vec3(C) * Sp - P;
            const float Face = dot(Dp, Dp) > 1e-8 ? 0.5 * (dot(normalize(Dp), N) + 1.0) : 1.0;
            const float Open = 1.0 - float((GP[b + 15u] >> 8u) & 255u) / 255.0;
            const float w = T.x * T.y * T.z * (Face * Face + 0.05) * (Open * Open + 1e-3);
            const vec3 c0 = vec3(uintBitsToFloat(GP[b]),      uintBitsToFloat(GP[b + 1u]),  uintBitsToFloat(GP[b + 2u]));
            const vec3 c1 = vec3(uintBitsToFloat(GP[b + 3u]), uintBitsToFloat(GP[b + 4u]),  uintBitsToFloat(GP[b + 5u]));
            const vec3 c2 = vec3(uintBitsToFloat(GP[b + 6u]), uintBitsToFloat(GP[b + 7u]),  uintBitsToFloat(GP[b + 8u]));
            const vec3 c3 = vec3(uintBitsToFloat(GP[b + 9u]), uintBitsToFloat(GP[b + 10u]), uintBitsToFloat(GP[b + 11u]));
            Sum += ProbeIrradiance(c0, c1, c2, c3, N) * w; W += w;
        }
        if (W > 1e-6) return Sum / W;
        G0 = GP[G0 + 11u];
        if (G0 == 0u) break;
    }
    return vec3(0.0);
}

void main()
{
    g_GIBvhTlasRoot = Push.m_TlasRoot; g_GIBvhPrimOffset = Push.m_PrimOffset; g_GIBvhSeeThrough = true;
    const uint lid = gl_LocalInvocationID.x;
    const uint p = GList[gl_WorkGroupID.x] % uint(max(Push.m_nProbes, 1));
    const uint  G0 = uint(Push.m_Base);
    const ivec3 Org = ivec3(int(GP[G0]), int(GP[G0 + 1u]), int(GP[G0 + 2u])), Dim = ivec3(int(GP[G0 + 4u]), int(GP[G0 + 5u]), int(GP[G0 + 6u]));
    const float Sp = uintBitsToFloat(GP[G0 + 3u]);
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
            vec3 In = Direct(Ph, Nh) + (Push.m_HitsReadProbes != 0 ? ProbesAt(Ph, Nh) : Indirect(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Q) ? Q : Unsupported(H.m_Inst, H.m_Tri, H.m_U, H.m_V));
            // (lighting shades the emitters' direct light - its emitter lights light what the probes light too -: the hit's own here, as Direct)
            if (EmitterDirectExternal()) { In += EmitterIrradianceAt(Ph, Nh, PHash(uint(Push.m_Frame) * 7919u + p * 64u + lid)); Emission = vec3(0); }
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
    const uint B = G0 + gi_probe_header_v + p * gi_probe_words_v;
    const bool bSame = (GP[B + 15u] & 1u) != 0u && ivec3(int(GP[B + 12u]), int(GP[B + 13u]), int(GP[B + 14u])) == Cell;
    const float a = bSame ? 0.2 : 1.0;                 // (a probe of another cell starts over)
    for (int k = 0; k < 4; ++k)
    {
        const vec3 Old = vec3(uintBitsToFloat(GP[B + 3u * uint(k)]), uintBitsToFloat(GP[B + 3u * uint(k) + 1u]), uintBitsToFloat(GP[B + 3u * uint(k) + 2u]));
        const vec3 New = mix(Old, s_SH[0][k], a);
        GP[B + 3u * uint(k)] = floatBitsToUint(New.x); GP[B + 3u * uint(k) + 1u] = floatBitsToUint(New.y); GP[B + 3u * uint(k) + 2u] = floatBitsToUint(New.z);
    }
    GP[B + 12u] = uint(Cell.x); GP[B + 13u] = uint(Cell.y); GP[B + 14u] = uint(Cell.z);
    const float Back = mix(bSame ? float((GP[B + 15u] >> 8u) & 255u) / 255.0 : 0.0, float(s_Back) / 64.0, a);
    GP[B + 15u] = 1u | (Back > 0.25 ? 2u : 0u) | (uint(Back * 255.0 + 0.5) << 8u);      // (a quarter of its rays on back faces: inside)
}
