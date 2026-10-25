#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require
// Part II, G2 (documentation/gi_dynamic_master_plan.md): a surfel pass on the GPU. Each work item's rays (gi::world::TraceChunk's) are
// traced by gi_lanes_v invocations, which write what TraceChunk accumulates before it finishes the pass (the light's sum, its share per basis
// direction, the depth moments' sums, the back-face and hit counts). Every function mirrors its CPU twin in xlionrender_surfel_gi.h; the
// data is gi::world::PackGPUSurfels's (its layout is written there) and PackGPU's
layout(local_size_x = 64) in;                                // (gi::world::gpu_group_v)
const int gi_lanes_v = 8;                                    // invocations per item (gi::world::gpu_lanes_v; the subgroup has 8 at least)

struct GIItem     { ivec4 m_Ids; vec4 m_Offsets; };          // instance, surfel, level, rays; off1, off2, shift (bits), -
struct GIResult   { float m_V[72]; };                         // sum 3, basis 9, moments 48, back, back all, unsupported, hits (int bits: gi::world::gpu_result), unsupported at 3, -
layout(std430, set = 0, binding = 5) readonly  buffer items_b     { GIItem     GIItems[]; };
layout(std430, set = 0, binding = 6) writeonly buffer results_b   { GIResult   GIResults[]; };
layout(std430, set = 0, binding = 7) readonly  buffer count_b     { uint       GICount[]; };    // the items there are (G4c: the expand's count; at most m_nItems)
layout(push_constant) uniform push_t { int m_nItems, m_TlasRoot, m_PrimOffset, m_First; } Push;     // (m_First: this dispatch's first item, a slice of the batch)
#include "xlionrender_gi_pass_common.glsl"

// A work item's rays are spread over gi_lanes_v invocations (a cluster of its subgroup): lane l traces rays l, l + 8, ... and the
// cluster adds what they found. One invocation per item traced its rays one after the other: a small batch was bound by the rays'
// latency (Police: 426 items, 3.4k rays in 5 ms)
void main()
{
    const int  k     = Push.m_First + int(gl_GlobalInvocationID.x) / gi_lanes_v;
    const int  Lane  = int(gl_GlobalInvocationID.x) % gi_lanes_v;
    // (an invocation past the items still joins its cluster's sums, with nothing; so does an item whose instance is gone or changed level
    // since the batch was picked: the world drops it)
    const int  nItems = min(Push.m_nItems, int(GICount[0]));
    const int  kk     = clamp(k, 0, max(nItems - 1, 0));
    const bool bItem = k < nItems && GD[InstRec(GIItems[kk].m_Ids.x) + 40u] != 0u && GD[InstRec(GIItems[kk].m_Ids.x) + 1u] == uint(GIItems[kk].m_Ids.z);
    g_GIBvhTlasRoot = Push.m_TlasRoot; g_GIBvhPrimOffset = Push.m_PrimOffset; g_GIBvhSeeThrough = true;
    vec3 Sum = vec3(0), BA0 = vec3(0), BA1 = vec3(0), BA2 = vec3(0), UnsAt = vec3(0);
    float MA[48]; for (int t = 0; t < 48; ++t) MA[t] = 0.0;
    int nBack = 0, nBackAll = 0, nUns = 0, nHits = 0, FirstUns = 0x7FFFFFFF;
    if (bItem)
    {
        const int Id = GIItems[k].m_Ids.x, s = GIItems[k].m_Ids.y, Rn = GIItems[k].m_Ids.w;
        const float Off1 = GIItems[k].m_Offsets.x, Off2 = GIItems[k].m_Offsets.y;
        const int Shift = floatBitsToInt(GIItems[k].m_Offsets.z);
        const uint R = InstRec(Id), M = MeshOf(R), Level = GD[R + 1u];
        const uint Sb = MSurfel(M, Level, s);
        const vec3 Sc = GV(Sb), Sn = GV(Sb + 3u);
        const float Reach = 2.0 * GF(Sb + 6u);
        const vec3 C = ToWorld(R, Sc), N = NormalToWorld(R, Sn);
        vec3 D0, D1, D2; BasisDirs(Sn, D0, D1, D2);
        const bool bSpread = GD[6] != 0u && Level == 0u && GD[M + 14u] != gi_none_v;
        const uint o0 = bSpread ? GD[GD[M + 14u] + uint(s)] : 0u, nO = bSpread ? GD[GD[M + 14u] + uint(s) + 1u] - o0 + 1u : 0u;     // (+1: its center)
        const float MeanScale = GF(R + 15u);
        for (int r = Lane; r < Rn; r += gi_lanes_v)
        {
            vec3 Cr = C, Nr = N;
            if (nO > 1u) { vec3 Po, No; GIOrigin(M, o0, uint(r + Shift) % nO, s, Po, No); Cr = ToWorld(R, Po); Nr = NormalToWorld(R, No); }
            const vec3 D = StratifiedCosineDir(Nr, r, Rn, Off1, Off2);
            const gi_hit H = GICast(Cr + Nr * 1e-3, D, 1e30);
            vec3 Dm = DirToMesh(R, D);
            const float Lm = length(Dm);
            Dm = Dm * (1.0 / Lm);
            const vec3 Pe = ToMesh(R, Cr + D * (H.m_Inst < 0 ? Reach / Lm : H.m_T)) - Sc;
            const float Le = length(Pe);
            const float dd = min(Le, Reach);
            const int t = MomentTexel(Sn, Le > 1e-6 ? Pe * (1.0 / Le) : Dm);
            MA[3 * t] += dd; MA[3 * t + 1] += dd * dd; MA[3 * t + 2] += 1.0;
            const vec3 Bw = BasisWeights(D0, D1, D2, Dm);
            if (H.m_Inst < 0) { const vec3 Sky = SkyAt(D); Sum += Sky; BA0 += Sky * Bw.x; BA1 += Sky * Bw.y; BA2 += Sky * Bw.z; continue; }
            const uint HR = InstRec(H.m_Inst), HM = MeshOf(HR);
            const uvec3 HT = MTri(HM, H.m_Tri);
            const float B0 = 1.0 - H.m_U - H.m_V;
            const vec3 Nh = NormalToWorld(HR, MNrm(HM, HT.x) * B0 + MNrm(HM, HT.y) * H.m_U + MNrm(HM, HT.z) * H.m_V);
            if (dot(Nh, D) > 0.0) { nBack += H.m_T < Reach * MeanScale ? 1 : 0; ++nBackAll; continue; }
            vec3 Albedo, Emission;
            SurfaceAt(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Albedo, Emission);
            vec3 Q = vec3(0.5);
            const bool bAblate = (GD[11] & 4u) != 0u;              // measurement only (setup::m_bAblateHitRead): the hit's cache read is a constant
            const bool bQ = bAblate || Indirect(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Q);
            const vec3 Ph = ToWorld(HR, MPos(HM, HT.x) * B0 + MPos(HM, HT.y) * H.m_U + MPos(HM, HT.z) * H.m_V);
            if (!bQ) { if (nUns == 0) { UnsAt = Ph; FirstUns = r; } ++nUns; }
            ++nHits;
            const vec3 In = Direct(Ph, Nh) + (bQ ? Q : (GD[11] & 32u) != 0u ? GV(HR + 5u) : Unsupported(H.m_Inst, H.m_Tri, H.m_U, H.m_V));     // (bit 32, measurement only: the fallback's walk replaced by the instance's mean)
            const vec3 Lr = Albedo * In * (1.0 / gi_pi_v) + (GD[13] == 0u && (GD[R + 47u] & 32u) == 0u ? Emission : vec3(0));     // (no light sampling, no links: the hits bring the emitters' light)
            Sum += Lr; BA0 += Lr * Bw.x; BA1 += Lr * Bw.y; BA2 += Lr * Bw.z;
        }
        // G5: the emitters' light by light sampling (gi::world::SampleEmitters: this lane's sample of gi_lanes_v): an emitter by weighted
        // reservoir sampling over the table (w lum(L), the same hashes), its shadow ray, L Q / lum(L) split into the basis as the links'
        if (GD[13] != 0u)
        {
            const uint Seed = Hash32(Hash32(floatBitsToUint(Off1) ^ Hash32(floatBitsToUint(Off2))) ^ uint(Lane + 1) * 0x9E3779B9u);
            const uint Tb = GD[14], nE = GD[15];
            float Q = 0.0; int Pick = -1;
            for (uint k = 0u; k < nE; ++k)
            {
                const uint e = Tb + 12u * k;
                if (Level == 0u && int(GD[e + 10u]) == Id && int(GD[e + 11u]) == s) continue;
                const vec3 D = GV(e) - C;
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
                const float u = float(Hash32(Seed ^ k * 0x85EBCA6Bu) >> 8) * (1.0 / 16777216.0);
                if (u * Q < q) Pick = int(k);
            }
            if (Pick >= 0)
            {
                const uint e = Tb + 12u * uint(Pick);
                const vec3 D = GV(e) - C;
                const float d = length(D);
                if (!Occluded(C + N * 1e-3, D * (1.0 / d), d * 0.98))
                {
                    const vec3 Le = GV(e + 7u), c = Le * (Q / (0.2126 * Le.x + 0.7152 * Le.y + 0.0722 * Le.z));
                    vec3 Dm = DirToMesh(R, D * (1.0 / d));
                    Dm = Dm * (1.0 / max(length(Dm), 1e-12));
                    const vec3 Bw = BasisWeights(D0, D1, D2, Dm);
                    const float f = float(Rn) / (3.0 * gi_pi_v);
                    const vec3 e0 = c * (3.0 * Bw.x / float(gi_lanes_v)) * f, e1 = c * (3.0 * Bw.y / float(gi_lanes_v)) * f, e2 = c * (3.0 * Bw.z / float(gi_lanes_v)) * f;
                    Sum += e0 + e1 + e2; BA0 += e0; BA1 += e1; BA2 += e2;
                }
            }
        }
    }
    // the cluster's sums (the first unsupported ray's hit: the lane with the item's lowest such ray)
    Sum = subgroupClusteredAdd(Sum, gi_lanes_v); BA0 = subgroupClusteredAdd(BA0, gi_lanes_v); BA1 = subgroupClusteredAdd(BA1, gi_lanes_v); BA2 = subgroupClusteredAdd(BA2, gi_lanes_v);
    for (int t = 0; t < 48; ++t) MA[t] = subgroupClusteredAdd(MA[t], gi_lanes_v);
    nBack = subgroupClusteredAdd(nBack, gi_lanes_v); nBackAll = subgroupClusteredAdd(nBackAll, gi_lanes_v);
    nUns = subgroupClusteredAdd(nUns, gi_lanes_v); nHits = subgroupClusteredAdd(nHits, gi_lanes_v);
    const int First = subgroupClusteredMin(FirstUns, gi_lanes_v);
    UnsAt = subgroupClusteredAdd(FirstUns == First && First != 0x7FFFFFFF ? UnsAt : vec3(0), gi_lanes_v);
    const ivec4 Work = subgroupClusteredAdd(ivec4(g_GIBvhVisits, g_Shadows, g_ReadEntries, g_Probes), gi_lanes_v);
    const int   nAccepted = subgroupClusteredAdd(g_Accepted, gi_lanes_v);
    if (!bItem || Lane != 0) return;
    GIResults[k].m_V[0] = Sum.x; GIResults[k].m_V[1] = Sum.y; GIResults[k].m_V[2] = Sum.z;
    GIResults[k].m_V[3] = BA0.x; GIResults[k].m_V[4] = BA0.y; GIResults[k].m_V[5] = BA0.z;
    GIResults[k].m_V[6] = BA1.x; GIResults[k].m_V[7] = BA1.y; GIResults[k].m_V[8] = BA1.z;
    GIResults[k].m_V[9] = BA2.x; GIResults[k].m_V[10] = BA2.y; GIResults[k].m_V[11] = BA2.z;
    for (int t = 0; t < 48; ++t) GIResults[k].m_V[12 + t] = MA[t];
    GIResults[k].m_V[60] = intBitsToFloat(nBack); GIResults[k].m_V[61] = intBitsToFloat(nBackAll); GIResults[k].m_V[62] = intBitsToFloat(nUns); GIResults[k].m_V[63] = intBitsToFloat(nHits);
    GIResults[k].m_V[64] = UnsAt.x; GIResults[k].m_V[65] = UnsAt.y; GIResults[k].m_V[66] = UnsAt.z; GIResults[k].m_V[67] = intBitsToFloat(nAccepted);
    for (int w = 0; w < 4; ++w) GIResults[k].m_V[68 + w] = intBitsToFloat(Work[w]);
}
