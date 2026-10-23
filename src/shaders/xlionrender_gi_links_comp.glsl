#version 450
// Experiment D (GI review: compiled transport), the links built on the GPU: world::CompileTransport's twin. One workgroup per surfel:
// its rays (the solver's cosine rays, origins and reads) are cast once; at each hit the read reports the surfels it blends and their
// weights (GI_EMIT, xlionrender_gi_pass_common.glsl), which are merged in shared memory, albedo weighted. The surfel keeps its
// gi_links_v heaviest links; the rest goes to a tail per instance (their mean light: world::TruncateTransport's tail through the
// instances), its gi_links_v heaviest kept. What no surfel carries (the sky, the direct light at the hits, the emission, the emitters'
// light the pass adds besides its rays) is its D. Level-0 instances only (a hit on a level-1 instance is dropped: counted)
layout(local_size_x = 64) in;

const uint gi_links_v  = 16u;                 // the links a surfel keeps, and the tail entries
const uint gi_slots_v  = 1024u;               // the shared table of the surfels its rays read
const uint gi_tslots_v = 64u;                 // the tail's table (per instance)
const uint gi_empty_v  = 0xFFFFFFFFu;
const float gi_fixed_v = 67108864.0;          // the weights' fixed point (2^26): integer atomics, the same sums in any order

struct GIItem { ivec4 m_Ids; vec4 m_Offsets; };     // instance, surfel, level, rays; off1, off2, shift (bits), -
layout(std430, set = 0, binding = 5) readonly  buffer items_b { GIItem GIItems[]; };
layout(std430, set = 0, binding = 6) writeonly buffer links_b { uvec4  GILinks[]; };     // per item 2 x gi_links_v: { key, r, g, b (float bits) }; key: instance << 16 | surfel (empty: ~0); the tail's: 1 << 31 | instance
layout(std430, set = 0, binding = 7) writeonly buffer d_b     { vec4   GILinkD[]; };     // per item: D, the dropped weight's luminance
layout(push_constant) uniform push_t { int m_nItems, m_TlasRoot, m_PrimOffset, m_First; } Push;

shared uint s_Key[gi_slots_v];
shared uint s_Val[gi_slots_v * 3u];
shared uint s_TKey[gi_tslots_v];
shared uint s_TVal[gi_tslots_v * 3u];
shared uint s_Dropped[3];
shared uint s_Best;
shared vec3 s_D[64];
vec3 g_CapColor = vec3(0);                    // the hit's albedo / rays (GI_EMIT multiplies it by the read's normalized weight)

void AddFixed(uint Base, vec3 c, bool bTail)
{
    const uvec3 u = uvec3(c * gi_fixed_v + 0.5);
    if (bTail) { atomicAdd(s_TVal[Base], u.x); atomicAdd(s_TVal[Base + 1u], u.y); atomicAdd(s_TVal[Base + 2u], u.z); }
    else       { atomicAdd(s_Val[Base],  u.x); atomicAdd(s_Val[Base + 1u],  u.y); atomicAdd(s_Val[Base + 2u],  u.z); }
}
void TailAdd(uint Inst, vec3 c)
{
    uint h = (Inst * 2654435761u) % gi_tslots_v;
    for (uint n = 0u; n < gi_tslots_v; ++n, h = (h + 1u) % gi_tslots_v)
    {
        const uint Prev = atomicCompSwap(s_TKey[h], gi_empty_v, Inst);
        if (Prev == gi_empty_v || Prev == Inst) { AddFixed(3u * h, c, true); return; }
    }
    const uvec3 u = uvec3(c * gi_fixed_v + 0.5);
    atomicAdd(s_Dropped[0], u.x); atomicAdd(s_Dropped[1], u.y); atomicAdd(s_Dropped[2], u.z);
}
void LinkEmit(int Inst, int s, float w)
{
    const vec3 c = g_CapColor * w;
    const uint Key = (uint(Inst) << 16) | uint(s);
    uint h = (Key * 2654435761u) % gi_slots_v;
    for (uint n = 0u; n < 64u; ++n, h = (h + 1u) % gi_slots_v)
    {
        const uint Prev = atomicCompSwap(s_Key[h], gi_empty_v, Key);
        if (Prev == gi_empty_v || Prev == Key) { AddFixed(3u * h, c, false); return; }
    }
    TailAdd(uint(Inst), c);                   // (the table full around it: the tail)
}
#define GI_EMIT(Inst, s, w) LinkEmit(Inst, s, w)
#include "xlionrender_gi_pass_common.glsl"

float Lum(vec3 c) { return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z; }
vec3  Unfix(uint a, uint b, uint c) { return vec3(float(a), float(b), float(c)) * (1.0 / gi_fixed_v); }

void main()
{
    const uint lid = gl_LocalInvocationID.x;
    const int  k   = Push.m_First + int(gl_WorkGroupID.x);
    for (uint j = lid; j < gi_slots_v; j += 64u) { s_Key[j] = gi_empty_v; s_Val[3u * j] = 0u; s_Val[3u * j + 1u] = 0u; s_Val[3u * j + 2u] = 0u; }
    if (lid < gi_tslots_v) { s_TKey[lid] = gi_empty_v; s_TVal[3u * lid] = 0u; s_TVal[3u * lid + 1u] = 0u; s_TVal[3u * lid + 2u] = 0u; }
    if (lid < 3u) s_Dropped[lid] = 0u;
    barrier();
    g_GIBvhTlasRoot = Push.m_TlasRoot; g_GIBvhPrimOffset = Push.m_PrimOffset; g_GIBvhSeeThrough = true;
    vec3 Dsum = vec3(0);
    const bool bItem = k < Push.m_nItems && GD[InstRec(GIItems[k].m_Ids.x) + 40u] != 0u && GD[InstRec(GIItems[k].m_Ids.x) + 1u] == 0u;
    if (bItem)
    {
        const int Id = GIItems[k].m_Ids.x, s = GIItems[k].m_Ids.y, Rn = GIItems[k].m_Ids.w;
        const float Off1 = GIItems[k].m_Offsets.x, Off2 = GIItems[k].m_Offsets.y;
        const uint R = InstRec(Id), M = MeshOf(R);
        const uint Sb = MSurfel(M, 0u, s);
        const vec3 Sc = GV(Sb), Sn = GV(Sb + 3u);
        const vec3 C = ToWorld(R, Sc), N = NormalToWorld(R, Sn);
        const bool bSpread = GD[6] != 0u && GD[M + 14u] != gi_none_v;
        const uint o0 = bSpread ? GD[GD[M + 14u] + uint(s)] : 0u, nO = bSpread ? GD[GD[M + 14u] + uint(s) + 1u] - o0 : 0u;
        for (int r = int(lid); r < Rn; r += 64)
        {
            vec3 Cr = C, Nr = N;
            if (nO > 1u) { const uint o = o0 + uint(r) % nO; Cr = ToWorld(R, GV(GD[M + 15u] + 3u * o)); Nr = NormalToWorld(R, GV(GD[M + 16u] + 3u * o)); }
            const vec3 D = StratifiedCosineDir(Nr, r, Rn, Off1, Off2);
            const gi_hit H = GICast(Cr + Nr * 1e-3, D, 1e30);
            if (H.m_Inst < 0) { Dsum += SkyAt(D) * (gi_pi_v / float(Rn)); continue; }
            const uint HR = InstRec(H.m_Inst), HM = MeshOf(HR);
            const uvec3 HT = MTri(HM, H.m_Tri);
            const float B0 = 1.0 - H.m_U - H.m_V;
            const vec3 Nh = NormalToWorld(HR, MNrm(HM, HT.x) * B0 + MNrm(HM, HT.y) * H.m_U + MNrm(HM, HT.z) * H.m_V);
            if (dot(Nh, D) > 0.0) continue;
            if (GD[HR + 1u] != 0u) continue;                           // (a level-1 instance: the prototype's rows are level 0)
            vec3 Albedo, Emission;
            SurfaceAt(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Albedo, Emission);
            const vec3 Ph = ToWorld(HR, MPos(HM, HT.x) * B0 + MPos(HM, HT.y) * H.m_U + MPos(HM, HT.z) * H.m_V);
            const vec3 Em = GD[13] == 0u && (GD[R + 47u] & 32u) == 0u ? Emission : vec3(0);
            Dsum += (Albedo * Direct(Ph, Nh) + Em * gi_pi_v) * (1.0 / float(Rn));
            vec3 Q;
            g_CapTier = 0u;
            if (Indirect(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Q))
            {
                g_CapColor = Albedo * (1.0 / float(Rn)); g_CapScale = 1.0 / g_ReadW; g_CapTier = g_ReadTier;
                Indirect(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Q);
                g_CapTier = 0u;
            }
            else
            {
                Unsupported(H.m_Inst, H.m_Tri, H.m_U, H.m_V);
                if (g_ReadTier == 4u)
                {
                    g_CapColor = Albedo * (1.0 / float(Rn)); g_CapScale = 1.0 / g_ReadW; g_CapTier = 4u;
                    Unsupported(H.m_Inst, H.m_Tri, H.m_U, H.m_V);
                    g_CapTier = 0u;
                }
            }
        }
        // the emitters' light the pass adds besides its rays (sampled: its expectation, every emitter in sight)
        if (GD[13] != 0u)
        {
            const uint Tb = GD[14], nE = GD[15];
            for (uint e8 = lid; e8 < nE; e8 += 64u)
            {
                const uint e = Tb + 12u * e8;
                if (int(GD[e + 10u]) == Id && int(GD[e + 11u]) == s) continue;
                const vec3 Dd = GV(e) - C;
                const float d2 = dot(Dd, Dd);
                if (d2 < 1e-10) continue;
                const float d = sqrt(d2);
                const vec3 Dn = Dd * (1.0 / d);
                const float cr = dot(N, Dn), ce = -dot(GV(e + 3u), Dn);
                if (cr <= 0.0 || ce <= 0.0) continue;
                const float A = GF(e + 6u), w = A * cr * ce / (d2 + A);
                if (w < 1e-5) continue;
                if (Occluded(C + N * 1e-3, Dn, d * 0.98)) continue;
                Dsum += GV(e + 7u) * w;
            }
        }
    }
    s_D[lid] = Dsum;
    barrier();
    // the gi_links_v heaviest: rounds of an atomic max over (luminance bits, slot)
    const uint Base = uint(max(k, 0)) * 2u * gi_links_v;
    for (uint q = 0u; q < gi_links_v; ++q)
    {
        if (lid == 0u) s_Best = 0u;
        barrier();
        uint Mine = 0u;
        for (uint j = lid; j < gi_slots_v; j += 64u)
            if (s_Key[j] != gi_empty_v && (s_Key[j] & 0x80000000u) == 0u)
            {
                const float l = Lum(Unfix(s_Val[3u * j], s_Val[3u * j + 1u], s_Val[3u * j + 2u]));
                Mine = max(Mine, (floatBitsToUint(max(l, 1e-30)) & ~(gi_slots_v - 1u)) | j);
            }
        atomicMax(s_Best, Mine);
        barrier();
        if (lid == 0u && bItem)
        {
            if (s_Best != 0u)
            {
                const uint j = s_Best & (gi_slots_v - 1u);
                GILinks[Base + q] = uvec4(s_Key[j], floatBitsToUint(Unfix(s_Val[3u * j], s_Val[3u * j + 1u], s_Val[3u * j + 2u])));
                s_Key[j] |= 0x80000000u;              // (taken)
            }
            else GILinks[Base + q] = uvec4(gi_empty_v, 0u, 0u, 0u);
        }
        barrier();
    }
    // the rest: the tail, per instance
    for (uint j = lid; j < gi_slots_v; j += 64u)
        if (s_Key[j] != gi_empty_v && (s_Key[j] & 0x80000000u) == 0u) TailAdd(s_Key[j] >> 16, Unfix(s_Val[3u * j], s_Val[3u * j + 1u], s_Val[3u * j + 2u]));
    barrier();
    for (uint q = 0u; q < gi_links_v; ++q)
    {
        if (lid == 0u) s_Best = 0u;
        barrier();
        uint Mine = 0u;
        if (lid < gi_tslots_v && s_TKey[lid] != gi_empty_v && (s_TKey[lid] & 0x80000000u) == 0u)
            Mine = (floatBitsToUint(max(Lum(Unfix(s_TVal[3u * lid], s_TVal[3u * lid + 1u], s_TVal[3u * lid + 2u])), 1e-30)) & ~(gi_tslots_v - 1u)) | lid;
        atomicMax(s_Best, Mine);
        barrier();
        if (lid == 0u && bItem)
        {
            if (s_Best != 0u)
            {
                const uint j = s_Best & (gi_tslots_v - 1u);
                GILinks[Base + gi_links_v + q] = uvec4(0x80000000u | s_TKey[j], floatBitsToUint(Unfix(s_TVal[3u * j], s_TVal[3u * j + 1u], s_TVal[3u * j + 2u])));
                s_TKey[j] |= 0x80000000u;
            }
            else GILinks[Base + gi_links_v + q] = uvec4(gi_empty_v, 0u, 0u, 0u);
        }
        barrier();
    }
    if (lid == 0u && bItem)
    {
        vec3 Dt = vec3(0);
        for (uint i = 0u; i < 64u; ++i) Dt += s_D[i];
        vec3 Drop = Unfix(s_Dropped[0], s_Dropped[1], s_Dropped[2]);
        for (uint j = 0u; j < gi_tslots_v; ++j) if (s_TKey[j] != gi_empty_v && (s_TKey[j] & 0x80000000u) == 0u) Drop += Unfix(s_TVal[3u * j], s_TVal[3u * j + 1u], s_TVal[3u * j + 2u]);
        GILinkD[k] = vec4(Dt, Lum(Drop));
    }
}
