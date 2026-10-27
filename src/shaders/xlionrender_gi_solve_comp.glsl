#version 450
// Experiment D (GI review: compiled transport), the solve on the GPU: world::SolveTransport's twin, one Jacobi iteration a frame over
// the links xlionrender_gi_links_comp.glsl built. Mode 0: per surfel E' = D + F (its first bounce of the lights) + sum of its links' weight x E + sum of its tail's weight x
// the instance's mean light (the means pass's: sum E V / sum V) into a scratch; mode 1: the scratch into the surfels' light (the
// world's E arrays: what the filter shows). A surfel whose instance is gone or not at level 0 is left alone
layout(local_size_x = 64) in;

struct GIItem { ivec4 m_Ids; vec4 m_Offsets; };
layout(std430, set = 0, binding = 0)          buffer data_b  { uint   GD[]; };
layout(std430, set = 0, binding = 1) readonly buffer links_b { uvec2  GILinks[]; };     // { key, weight (RGB9E5) }
layout(std430, set = 0, binding = 2) readonly buffer d_b     { vec4   GILinkD[]; };
layout(std430, set = 0, binding = 3) readonly buffer items_b { GIItem GIItems[]; };
layout(std430, set = 0, binding = 4) readonly buffer means_b { float  GIMeans[]; };
layout(std430, set = 0, binding = 5)          buffer next_b  { vec4   GINext[]; };
layout(std430, set = 0, binding = 6) readonly buffer dir_b   { vec4   GIDir[]; };       // per item its first bounce of the lights, F (the links' mode 1)
layout(push_constant) uniform push_t { int m_nItems, m_Mode, m_nInst, m_Links; } Push;

const uint gi_inst_v  = 49u;
const uint gi_empty_v = 0xFFFFFFFFu;

uint InstRec(uint i) { return GD[0] + i * gi_inst_v; }
bool Live(uint i)    { const uint R = InstRec(i); return GD[R + 40u] != 0u && GD[R + 1u] == 0u; }
vec3 UnpackRGB9E5(uint p) { const float s = uintBitsToFloat(uint(127 + int(p >> 27) - 24) << 23); return vec3(float(p & 511u), float((p >> 9) & 511u), float((p >> 18) & 511u)) * s; }
vec3 E(uint i, uint s) { const uint a = GD[InstRec(i) + 2u] + 3u * s; return vec3(uintBitsToFloat(GD[a]), uintBitsToFloat(GD[a + 1u]), uintBitsToFloat(GD[a + 2u])); }


void main()
{
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
    vec3 Sum = GILinkD[k].xyz + GIDir[k].xyz;
    const uint Base = k * 2u * uint(Push.m_Links);
    for (uint q = 0u; q < 2u * uint(Push.m_Links); ++q)
    {
        const uvec2 L = GILinks[Base + q];
        if (L.x == gi_empty_v) continue;
        const vec3 w = UnpackRGB9E5(L.y);
        if ((L.x & 0x80000000u) != 0u)
        {
            const uint j = L.x & 0x7FFFFFFFu;
            if (j >= uint(Push.m_nInst)) continue;
            const float W = GIMeans[5u * j + 3u];
            if (W > 1e-6) Sum += w * vec3(GIMeans[5u * j], GIMeans[5u * j + 1u], GIMeans[5u * j + 2u]) * (1.0 / W);
        }
        else
        {
            const uint j = L.x >> 16;
            if (j < uint(Push.m_nInst) && Live(j)) Sum += w * E(j, L.x & 0xFFFFu);
        }
    }
    GINext[k] = vec4(Sum, 0.0);
}
