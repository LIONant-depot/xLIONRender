#version 450
// GI sections (documentation/gi_chunks_plan.md, "Owners and ghosts"): a section's ghost slots are copies of other sections' surfels near its
// border. After the commit, every ghost takes its owner's published set - its light, its validity, its depth moments, its basis fractions
// (gi::world::SyncGhosts, the CPU's twin): the reads, the filter and the means after it read the same epoch for owned and ghost surfels.
// One invocation a ghost: { the ghost's instance, its surfel, the owner's instance, its surfel }. Equal levels only (both level 0: a
// section at another level keeps the ghost's last set - the plan's gate)
layout(local_size_x = 64) in;
layout(std430, set = 0, binding = 0)          buffer data_b   { uint  GD[]; };
layout(std430, set = 0, binding = 1) readonly buffer ghosts_b { uvec4 GIGhosts[]; };
layout(push_constant) uniform push_t { int m_nGhosts, m_Pad0, m_Pad1, m_Pad2; } Push;

const uint gi_inst_v = 49u;

void main()
{
    const uint i = gl_GlobalInvocationID.x;
    if (i >= uint(Push.m_nGhosts)) return;
    const uvec4 G = GIGhosts[i];
    const uint Rd = GD[0] + G.x * gi_inst_v, Rs = GD[0] + G.z * gi_inst_v;
    if (GD[Rd + 40u] == 0u || GD[Rs + 40u] == 0u || GD[Rd + 1u] != 0u || GD[Rs + 1u] != 0u) return;     // (not alive, or not both fine)
    const uint d = G.y, s = G.w;
    for (uint k = 0u; k < 3u; ++k)  GD[GD[Rd + 2u] + 3u * d + k]  = GD[GD[Rs + 2u] + 3u * s + k];      // light
    GD[GD[Rd + 3u] + d] = GD[GD[Rs + 3u] + s];                                                              // validity
    for (uint k = 0u; k < 32u; ++k) GD[GD[Rd + 4u] + 32u * d + k] = GD[GD[Rs + 4u] + 32u * s + k];     // depth moments
    for (uint k = 0u; k < 9u; ++k)  GD[GD[Rd + 41u] + 9u * d + k] = GD[GD[Rs + 41u] + 9u * s + k];     // basis fractions
}
