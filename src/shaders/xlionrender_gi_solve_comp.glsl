#version 450
// Experiment D (GI review: compiled transport), the solve on the GPU: world::SolveTransport's twin, one Jacobi iteration a frame over
// the links xlionrender_gi_links_comp.glsl built. Mode 0: per surfel E' = D + sum of its links' weight x E + sum of its tail's weight x
// the instance's mean light (the means pass's: sum E V / sum V) into a scratch; mode 1: the scratch into the surfels' light (the
// world's E arrays: what the filter shows). A surfel whose instance is gone or not at level 0 is left alone
layout(local_size_x = 64) in;

struct GIItem { ivec4 m_Ids; vec4 m_Offsets; };
layout(std430, set = 0, binding = 0)          buffer data_b  { uint   GD[]; };
layout(std430, set = 0, binding = 1) readonly buffer links_b { uvec4  GILinks[]; };
layout(std430, set = 0, binding = 2) readonly buffer d_b     { vec4   GILinkD[]; };
layout(std430, set = 0, binding = 3) readonly buffer items_b { GIItem GIItems[]; };
layout(std430, set = 0, binding = 4) readonly buffer means_b { float  GIMeans[]; };
layout(std430, set = 0, binding = 5)          buffer next_b  { vec4   GINext[]; };
layout(std430, set = 0, binding = 6) readonly buffer dir_b   { vec4   GIDir[]; };       // per item its direct light (the links' mode 1)
layout(std430, set = 0, binding = 7) readonly buffer first_b { uint   GIInstFirst[]; }; // per instance its first item (+1: the total)
layout(std430, set = 0, binding = 8)          buffer dmean_b { vec4   GIDirMean[]; };   // per instance its items' mean direct light (mode 2)
layout(std430, set = 0, binding = 9)          buffer dir0_b  { vec4   GIDir0[]; };      // per item its direct light when the links were built (D has it)
layout(std430, set = 0, binding = 10)         buffer dmean0_b { vec4  GIDirMean0[]; }; // per instance the same, its mean
layout(std430, set = 0, binding = 11) readonly buffer list_b  { uint   GIList[]; };      // mode 3: the items just built again
shared vec3 s_Sum[64];
layout(push_constant) uniform push_t { int m_nItems, m_Mode, m_nInst, m_Links; } Push;

const uint gi_inst_v  = 48u;
const uint gi_empty_v = 0xFFFFFFFFu;

uint InstRec(uint i) { return GD[0] + i * gi_inst_v; }
bool Live(uint i)    { const uint R = InstRec(i); return GD[R + 40u] != 0u && GD[R + 1u] == 0u; }
vec3 E(uint i, uint s) { const uint a = GD[InstRec(i) + 2u] + 3u * s; return vec3(uintBitsToFloat(GD[a]), uintBitsToFloat(GD[a + 1u]), uintBitsToFloat(GD[a + 2u])); }

// a link's surfel: its light and how its direct light changed since the build (D has the build's: E = D + T (E + Dir - Dir0))
vec3 EDir(uint i, uint s) { const uint k = GIInstFirst[i] + s; return E(i, s) + GIDir[k].xyz - GIDir0[k].xyz; }

void main()
{
    if (Push.m_Mode == 3)           // the items built again (a local rebuild): their D has the direct light of now - so has their Dir0
    {
        const uint j = gl_GlobalInvocationID.x;
        if (j < uint(Push.m_nItems)) GIDir0[GIList[j]] = GIDir[GIList[j]];
        return;
    }
    if (Push.m_Mode == 2 || Push.m_Mode == 4)       // per instance (one workgroup each) its items' mean direct light: the tail's (4: of Dir0)
    {
        const uint i = gl_WorkGroupID.x, lid = gl_LocalInvocationID.x;
        if (i >= uint(Push.m_nInst)) return;
        const uint a = GIInstFirst[i], b = GIInstFirst[i + 1u];
        vec3 S = vec3(0);
        for (uint j = a + lid; j < b; j += 64u) S += Push.m_Mode == 4 ? GIDir0[j].xyz : GIDir[j].xyz;
        s_Sum[lid] = S;
        barrier();
        if (lid == 0u)
        {
            vec3 T = vec3(0);
            for (uint j = 0u; j < 64u; ++j) T += s_Sum[j];
            const vec4 Mn = vec4(b > a ? T / float(b - a) : vec3(0), 0.0);
            if (Push.m_Mode == 4) GIDirMean0[i] = Mn; else GIDirMean[i] = Mn;
        }
        return;
    }
    const uint k = gl_GlobalInvocationID.x;
    if (k >= uint(Push.m_nItems)) return;
    const uint Inst = uint(GIItems[k].m_Ids.x), s = uint(GIItems[k].m_Ids.y);
    if (uint(Inst) >= uint(Push.m_nInst) || !Live(Inst)) return;
    if (Push.m_Mode == 1)
    {
        const uint a = GD[InstRec(Inst) + 2u] + 3u * s;
        GD[a] = floatBitsToUint(GINext[k].x); GD[a + 1u] = floatBitsToUint(GINext[k].y); GD[a + 2u] = floatBitsToUint(GINext[k].z);
        return;
    }
    vec3 Sum = GILinkD[k].xyz;
    const uint Base = k * 2u * uint(Push.m_Links);
    for (uint q = 0u; q < 2u * uint(Push.m_Links); ++q)
    {
        const uvec4 L = GILinks[Base + q];
        if (L.x == gi_empty_v) continue;
        const vec3 w = uintBitsToFloat(L.yzw);
        if ((L.x & 0x80000000u) != 0u)
        {
            const uint j = L.x & 0x7FFFFFFFu;
            if (j >= uint(Push.m_nInst)) continue;
            const float W = GIMeans[5u * j + 3u];
            if (W > 1e-6) Sum += w * (vec3(GIMeans[5u * j], GIMeans[5u * j + 1u], GIMeans[5u * j + 2u]) * (1.0 / W) + GIDirMean[j].xyz - GIDirMean0[j].xyz);
        }
        else
        {
            const uint j = L.x >> 16;
            if (j < uint(Push.m_nInst) && Live(j)) Sum += w * EDir(j, L.x & 0xFFFFu);
        }
    }
    GINext[k] = vec4(Sum, 0.0);
}
