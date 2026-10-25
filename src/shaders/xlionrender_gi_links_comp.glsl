#version 450
// Experiment D (GI review: compiled transport), the links built on the GPU: world::CompileTransport's twin. One workgroup per surfel:
// its rays (the solver's cosine rays, origins and reads) are cast once; at each hit the read reports the surfels it blends and their
// weights (GI_EMIT, xlionrender_gi_pass_common.glsl), which are merged in shared memory, albedo weighted. The surfel keeps its
// gi_links_v heaviest links; the rest goes to a tail per instance (their mean light: world::TruncateTransport's tail through the
// instances), its gi_links_v heaviest kept. What no surfel carries (the sky, the direct light at the hits, the emission, the emitters'
// light the pass adds besides its rays) is its D. Level-0 instances only (a hit on a level-1 instance is dropped: counted)
// Relightable: no light is in D (the sky and the emission only). Each item keeps its rays' hits as gi_hits_v clusters (position,
// normal, albedo / rays): its first bounce of the lights, F = sum albedo x Direct(hit), is made again from them whenever a light changes
// (mode 1) - E = D + F + T E. The surfels' own direct light read through the links instead (one point each, through 16 links and an
// instance's mean) was 32-37% off per surfel on the Police level; the hits' baked and relit by a difference left a moved lamp's spot lit.
// Mode 1: one invocation per item, its F (into GIDirTmp). Mode 4: one invocation per item, F with its mesh neighbors' (the display
// filter's lists and weights): each item picks one hit a cluster, its own noise (15% a surfel on the Police level); the light is smooth. Mode 2: the build of the items GIList names (from m_First: a workgroup each) - what a moved
// object or a changed material touches (gpu_trace::PumpLinks' local rebuild). Mode 3: one invocation per item, flagged when it or one of
// its links or its tail reads an instance GIChanged names
layout(local_size_x = 64) in;

const uint gi_links_v  = 16u;                 // the links a surfel keeps, and the tail entries
const uint gi_slots_v  = 1024u;               // the shared table of the surfels its rays read
const uint gi_tslots_v = 64u;                 // the tail's table (per instance)
const uint gi_empty_v  = 0xFFFFFFFFu;
const uint gi_hits_v   = 8u;                  // the hit clusters an item keeps (its first bounce)
const uint gi_hitcap_v = 256u;                // the hits clustered (its rays' first 256)
const float gi_fixed_v = 67108864.0;          // the weights' fixed point (2^26): integer atomics, the same sums in any order

struct GIItem { ivec4 m_Ids; vec4 m_Offsets; };     // instance, surfel, level, rays; off1, off2, shift (bits), -
layout(std430, set = 0, binding = 5) readonly  buffer items_b { GIItem GIItems[]; };
layout(std430, set = 0, binding = 6)           buffer links_b { uvec2  GILinks[]; };     // per item 2 x gi_links_v: { key, weight (RGB9E5) }; key: instance << 16 | surfel (empty: ~0); the tail's: 1 << 31 | instance
layout(std430, set = 0, binding = 7) writeonly buffer d_b     { vec4   GILinkD[]; };     // per item: D, the dropped weight's luminance
layout(std430, set = 0, binding = 8)           buffer dir_b   { vec4   GIDir[]; };       // per item: its first bounce of the lights, F (mode 4: smoothed)
layout(std430, set = 0, binding = 9)  readonly buffer list_b  { uint   GIList[]; };      // mode 2: the items to build
layout(std430, set = 0, binding = 10) readonly buffer chg_b   { uint   GIChanged[]; };   // mode 3: per instance 1 when it changed
layout(std430, set = 0, binding = 11) writeonly buffer flag_b { uint   GIFlag[]; };      // mode 3: per item 1 when the change touches it
layout(std430, set = 0, binding = 12)           buffer hits_b { uvec4  GIHits[]; };
layout(std430, set = 0, binding = 13) readonly  buffer first_b { uint  GIInstFirst[]; };  // per instance its first item (+1: the total)
layout(std430, set = 0, binding = 14)           buffer ftmp_b  { vec4  GIDirTmp[]; };     // mode 1: F as its clusters make it; mode 4 smooths it into GIDir      // per item gi_hits_v: { half2 offset xy, half2 (offset z, albedo r), half2 albedo gb, snorm2 octahedral normal }
layout(push_constant) uniform push_t { int m_nItems, m_TlasRoot, m_PrimOffset, m_First, m_Mode, m_Pad0, m_Pad1, m_Pad2; } Push;

shared uint s_Key[gi_slots_v];
shared uint s_Val[gi_slots_v * 3u];
shared uint s_TKey[gi_tslots_v];
shared uint s_TVal[gi_tslots_v * 3u];
shared uint s_Dropped[3];
shared uint s_Best;
shared vec3 s_D[64];
shared vec3 s_HP[gi_hitcap_v], s_HN[gi_hitcap_v], s_HA[gi_hitcap_v];   // the hits: offset from the item's center, normal, albedo / rays
shared uint s_HC[gi_hitcap_v];                                          // their cluster (gi_empty_v: no hit there)
shared uint s_nH;                                                       // the rays (their slots: the first gi_hitcap_v)
shared vec3 s_CP[gi_hits_v], s_CN[gi_hits_v];
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
// gi::PackRGB9E5, step for step (the same bits): a link's weight in 32 bits (9 bits a channel, a shared exponent)
uint PackRGB9E5(vec3 c)
{
    c = vec3(c.x > 0.0 ? min(c.x, 65408.0) : 0.0, c.y > 0.0 ? min(c.y, 65408.0) : 0.0, c.z > 0.0 ? min(c.z, 65408.0) : 0.0);
    const float m = max(c.x, max(c.y, c.z));
    const int Floor = m > 0.0 ? int((floatBitsToUint(m) >> 23) & 0xFFu) - 127 : -16;
    int e = max(-16, Floor) + 16;
    float s = uintBitsToFloat(uint(127 - (e - 24)) << 23);
    if (floor(m * s + 0.5) >= 512.0) { ++e; s *= 0.5; }
    const uvec3 M = uvec3(floor(c * s + 0.5));
    return M.x | (M.y << 9) | (M.z << 18) | (uint(e) << 27);
}
vec2 OctEnc(vec3 n) { n /= abs(n.x) + abs(n.y) + abs(n.z); vec2 e = n.xy; if (n.z < 0.0) e = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0); return e; }
vec3 OctDec(vec2 e) { vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y)); if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0); return normalize(n); }
vec3  Unfix(uint a, uint b, uint c) { return vec3(float(a), float(b), float(c)) * (1.0 / gi_fixed_v); }

void main()
{
    g_GIBvhTlasRoot = Push.m_TlasRoot; g_GIBvhPrimOffset = Push.m_PrimOffset; g_GIBvhSeeThrough = true;
    if (Push.m_Mode == 3)           // the items a change touches: its own instance changed, or a link or its tail reads one that did
    {
        const int kf = int(gl_GlobalInvocationID.x);
        if (kf >= Push.m_nItems) return;
        bool bHit = GIChanged[uint(GIItems[kf].m_Ids.x)] != 0u;
        const uint Bf = uint(kf) * 2u * gi_links_v;
        for (uint q = 0u; q < 2u * gi_links_v && !bHit; ++q)
        {
            const uint Kq = GILinks[Bf + q].x;
            if (Kq == gi_empty_v) continue;
            bHit = GIChanged[(Kq & 0x80000000u) != 0u ? (Kq & 0x7FFFFFFFu) : (Kq >> 16)] != 0u;
        }
        GIFlag[kf] = bHit ? 1u : 0u;
        return;
    }
    if (Push.m_Mode == 1)           // the direct light at each item's surfel (world::Direct at its center): what its links' weights carry
    {
        const int kd = Push.m_First + int(gl_GlobalInvocationID.x);
        if (kd >= Push.m_nItems) return;
        const int Id = GIItems[kd].m_Ids.x, sd = GIItems[kd].m_Ids.y;
        const uint Rd = InstRec(Id);
        if (GD[Rd + 40u] == 0u || GD[Rd + 1u] != 0u) { GIDir[kd] = vec4(0.0); return; }
        const uint Sd = MSurfel(MeshOf(Rd), 0u, sd);
        const vec3 Cd = ToWorld(Rd, GV(Sd));
        vec3 F = vec3(0);
        for (uint q = 0u; q < gi_hits_v; ++q)
        {
            const uvec4 h = GIHits[uint(kd) * gi_hits_v + q];
            const vec2 xy = unpackHalf2x16(h.x), zr = unpackHalf2x16(h.y), gb = unpackHalf2x16(h.z);
            const vec3 A = vec3(zr.y, gb);
            if (A.x + A.y + A.z <= 0.0) continue;
            const vec3 Nh = OctDec(unpackSnorm2x16(h.w));
            F += A * Direct(Cd + vec3(xy, zr.x), Nh);
        }
        GIDirTmp[kd] = vec4(F, 0.0);
        return;
    }
    if (Push.m_Mode == 4)
    {
        const int kd = Push.m_First + int(gl_GlobalInvocationID.x);
        if (kd >= Push.m_nItems) return;
        const int Id = GIItems[kd].m_Ids.x, sd = GIItems[kd].m_Ids.y;
        const uint Rd = InstRec(Id), Md = MeshOf(Rd), Nf = GD[Md + 18u];
        vec3 F = GIDirTmp[kd].xyz; float W = 1.0;
        if (Nf != gi_none_v && GD[Rd + 40u] != 0u && GD[Rd + 1u] == 0u)
        {
            const uint Ni = GD[Md + 19u], Nw = GD[Md + 20u], First = GIInstFirst[uint(Id)];
            for (uint q = GD[Nf + uint(sd)], q1 = GD[Nf + uint(sd) + 1u]; q < q1; ++q)
            {
                const float w = GF(Nw + q);
                F += GIDirTmp[First + GD[Ni + q]].xyz * w; W += w;
            }
        }
        GIDir[kd] = vec4(F / W, 0.0);
        return;
    }
    const uint lid = gl_LocalInvocationID.x;
    const int  k   = Push.m_Mode == 2 ? int(GIList[uint(Push.m_First) + gl_WorkGroupID.x]) : Push.m_First + int(gl_WorkGroupID.x);
    for (uint j = lid; j < gi_slots_v; j += 64u) { s_Key[j] = gi_empty_v; s_Val[3u * j] = 0u; s_Val[3u * j + 1u] = 0u; s_Val[3u * j + 2u] = 0u; }
    if (lid < gi_tslots_v) { s_TKey[lid] = gi_empty_v; s_TVal[3u * lid] = 0u; s_TVal[3u * lid + 1u] = 0u; s_TVal[3u * lid + 2u] = 0u; }
    if (lid < 3u) s_Dropped[lid] = 0u;
    for (uint j = lid; j < gi_hitcap_v; j += 64u) s_HC[j] = gi_empty_v;
    if (lid == 0u) s_nH = 0u;
    barrier();
    vec3 Dsum = vec3(0);
    vec3 C = vec3(0);
    const bool bItem = k < Push.m_nItems && GD[InstRec(GIItems[k].m_Ids.x) + 40u] != 0u && GD[InstRec(GIItems[k].m_Ids.x) + 1u] == 0u;
    if (bItem)
    {
        const int Id = GIItems[k].m_Ids.x, s = GIItems[k].m_Ids.y, Rn = GIItems[k].m_Ids.w;
        const float Off1 = GIItems[k].m_Offsets.x, Off2 = GIItems[k].m_Offsets.y;
        const uint R = InstRec(Id), M = MeshOf(R);
        const uint Sb = MSurfel(M, 0u, s);
        const vec3 Sc = GV(Sb), Sn = GV(Sb + 3u);
        C = ToWorld(R, Sc);
        const vec3 N = NormalToWorld(R, Sn);
        const bool bSpread = GD[6] != 0u && GD[M + 14u] != gi_none_v;
        const uint o0 = bSpread ? GD[GD[M + 14u] + uint(s)] : 0u, nO = bSpread ? GD[GD[M + 14u] + uint(s) + 1u] - o0 + 1u : 0u;     // (+1: its center)
        for (int r = int(lid); r < Rn; r += 64)
        {
            vec3 Cr = C, Nr = N;
            if (nO > 1u) { vec3 Po, No; GIOrigin(M, o0, uint(r) % nO, s, Po, No); Cr = ToWorld(R, Po); Nr = NormalToWorld(R, No); }
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
            Dsum += Em * (gi_pi_v / float(Rn));
            if (uint(r) < gi_hitcap_v) { s_HP[r] = Ph - C; s_HN[r] = Nh; s_HA[r] = Albedo * (1.0 / float(Rn)); s_HC[r] = 0u; }
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
    if (lid == 0u && bItem) s_nH = uint(GIItems[k].m_Ids.w);
    barrier();
    // its hits in gi_hits_v clusters (k-means: by position, normals apart by a surfel's reach), their albedo summed: what mode 1 lights;
    // each lit at one of its hits picked by albedo (a stratum: the light through its mean point was 10% over in the Cornell box at 8
    // clusters - lit and shadowed hits merged). The hits by their ray (slots in ray order: the same clusters every build)
    {
        const uint nS = min(s_nH, gi_hitcap_v);
        uint nV = 0u;                                 // (the valid slots, the clusters' seeds evenly among them)
        for (uint i = 0u; i < nS; ++i) if (s_HC[i] != gi_empty_v) ++nV;
        const uint nC = min(gi_hits_v, nV);
        if (lid < nC)
        {
            const uint Want = (lid * nV) / gi_hits_v;
            for (uint i = 0u, v = 0u; i < nS; ++i) if (s_HC[i] != gi_empty_v) { if (v == Want) { s_CP[lid] = s_HP[i]; s_CN[lid] = s_HN[i]; break; } ++v; }
        }
        barrier();
        float Scale = 0.0;                            // (the normals' weight: the hits' spread around their first, squared)
        vec3 P0 = vec3(0);
        for (uint i = 0u; i < nS; ++i) if (s_HC[i] != gi_empty_v) { P0 = s_HP[i]; break; }
        for (uint i = 0u; i < nS; ++i) if (s_HC[i] != gi_empty_v) Scale += dot(s_HP[i] - P0, s_HP[i] - P0);
        Scale = nV > 0u ? 4.0 * Scale / float(nV) + 1e-6 : 1.0;
        for (int it = 0; it < 4; ++it)
        {
            for (uint i = lid; i < nS; i += 64u)
            {
                if (s_HC[i] == gi_empty_v) continue;
                uint Best = 0u; float Bd = 1e30;
                for (uint q = 0u; q < nC; ++q)
                {
                    const vec3 d = s_HP[i] - s_CP[q];
                    const float e = dot(d, d) + Scale * (1.0 - dot(s_HN[i], s_CN[q]));
                    if (e < Bd) { Bd = e; Best = q; }
                }
                s_HC[i] = Best;
            }
            barrier();
            if (lid < nC)
            {
                vec3 P = vec3(0), Nn = vec3(0); float W = 0.0;
                for (uint i = 0u; i < nS; ++i) if (s_HC[i] == lid) { const float w = Lum(s_HA[i]) + 1e-9; P += s_HP[i] * w; Nn += s_HN[i] * w; W += w; }
                if (W > 0.0) { s_CP[lid] = P / W; s_CN[lid] = length(Nn) > 1e-12 ? normalize(Nn) : s_CN[lid]; }
            }
            barrier();
        }
        if (lid < gi_hits_v && bItem)
        {
            vec3 A = vec3(0); float Tot = 0.0;
            for (uint i = 0u; i < nS; ++i) if (s_HC[i] == lid) { A += s_HA[i]; Tot += Lum(s_HA[i]) + 1e-9; }
            if (lid >= nC) A = vec3(0);
            if (s_nH > gi_hitcap_v) A *= float(s_nH) / float(gi_hitcap_v);    // (the rays past the slots: their share scaled in)
            const float u = fract(float((uint(k) * 2654435761u) ^ (lid * 2246822519u)) * (1.0 / 4294967296.0)) * Tot;
            uint Pick = gi_empty_v; float Acc = 0.0;
            for (uint i = 0u; i < nS; ++i) if (s_HC[i] == lid) { Pick = i; Acc += Lum(s_HA[i]) + 1e-9; if (Acc >= u) break; }
            const vec3 P = Pick != gi_empty_v ? s_HP[Pick] : vec3(0), Nh = Pick != gi_empty_v ? s_HN[Pick] : vec3(0, 0, 1);
            GIHits[uint(k) * gi_hits_v + lid] = uvec4(packHalf2x16(P.xy), packHalf2x16(vec2(P.z, A.x)), packHalf2x16(A.yz), packSnorm2x16(OctEnc(Nh)));
        }
    }
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
                GILinks[Base + q] = uvec2(s_Key[j], PackRGB9E5(Unfix(s_Val[3u * j], s_Val[3u * j + 1u], s_Val[3u * j + 2u])));
                s_Key[j] |= 0x80000000u;              // (taken)
            }
            else GILinks[Base + q] = uvec2(gi_empty_v, 0u);
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
                GILinks[Base + gi_links_v + q] = uvec2(0x80000000u | s_TKey[j], PackRGB9E5(Unfix(s_TVal[3u * j], s_TVal[3u * j + 1u], s_TVal[3u * j + 2u])));
                s_TKey[j] |= 0x80000000u;
            }
            else GILinks[Base + gi_links_v + q] = uvec2(gi_empty_v, 0u);
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
