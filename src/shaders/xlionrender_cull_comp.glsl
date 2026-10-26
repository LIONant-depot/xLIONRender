#version 450
// GPU cluster culling (documentation/gpu_cluster_culling.md, features/cull/xlionrender_gpu_cull.h): a thread a candidate cluster. Its
// local box through its draw's local to clip; in the view, a command and a record appended to its run (and to its prepass run): the
// scene pass draws each run with one DrawIndexedIndirectCount (the record: xgeom_static_mb_clusters.vert XLION_RECORDS, by firstInstance)
layout(local_size_x = 64) in;

struct candidate
{
    uvec4   Runs;           // its run, the run's first command, its prepass run (0xFFFFFFFF: none), that run's first command
    uvec4   Draw;           // its draw's matrix, the command: index count, first index, vertex offset
    uvec4   Rec0;           // the record: cluster, instance, GI offset, GI level
    uvec4   Rec1;           //   material base, GI vertex delta
    vec4    Center;         // its box (local)
    vec4    Half;
};
layout(std430, set = 0, binding = 0) readonly  buffer cand_b    { candidate C[]; };
layout(std430, set = 0, binding = 1) readonly  buffer mat_b     { mat4 L2C[]; };
layout(std430, set = 0, binding = 2)           buffer count_b   { uint Counts[]; };
layout(std430, set = 0, binding = 3) writeonly buffer cmd_b     { uint Cmd[]; };        // VkDrawIndexedIndirectCommand: 5 each
layout(std430, set = 0, binding = 4) writeonly buffer rec_b     { uint Rec[]; };        // the vertex shader's DrawRecord: 6 each

layout(push_constant) uniform push_t { uint m_First, m_Count, m_MatBase, m_Pad; } Push;    // this frame's candidates and matrices

void Append(uint Run, uint First, candidate K, uvec4 R0, uvec2 R1)
{
    const uint o = First + atomicAdd(Counts[Run], 1u);
    Cmd[o * 5u + 0u] = K.Draw.y; Cmd[o * 5u + 1u] = 1u; Cmd[o * 5u + 2u] = K.Draw.z; Cmd[o * 5u + 3u] = K.Draw.w; Cmd[o * 5u + 4u] = o;
    Rec[o * 6u + 0u] = R0.x; Rec[o * 6u + 1u] = R0.y; Rec[o * 6u + 2u] = R0.z; Rec[o * 6u + 3u] = R0.w; Rec[o * 6u + 4u] = R1.x; Rec[o * 6u + 5u] = R1.y;
}

void main()
{
    const uint i = gl_GlobalInvocationID.x;
    if (i >= Push.m_Count) return;
    const candidate K = C[Push.m_First + i];
    const mat4 M = L2C[Push.m_MatBase + K.Draw.x];

    // The box in clip space: its center and half axes; outside a clip plane P when its nearest corner is: dot(P, center) + sum |dot(P, axis)| < 0
    const vec4 Cc = M * vec4(K.Center.xyz, 1.0);
    const vec4 Ax = M[0] * K.Half.x, Ay = M[1] * K.Half.y, Az = M[2] * K.Half.z;
    const vec4 Planes[6] = vec4[6](vec4(1, 0, 0, 1), vec4(-1, 0, 0, 1), vec4(0, 1, 0, 1), vec4(0, -1, 0, 1), vec4(0, 0, 1, 0), vec4(0, 0, -1, 1));   // Vulkan's clip volume: z in [0, w]
    for (int p = 0; p < 6; ++p)
        if (dot(Planes[p], Cc) + abs(dot(Planes[p], Ax)) + abs(dot(Planes[p], Ay)) + abs(dot(Planes[p], Az)) < 0.0) return;

    Append(K.Runs.x, K.Runs.y, K, K.Rec0, K.Rec1.xy);
    if (K.Runs.z != 0xFFFFFFFFu) Append(K.Runs.z, K.Runs.w, K, uvec4(K.Rec0.xy, 0u, 0xFFFFFFFFu), uvec2(0u, 0u));     // (the prepass: no GI, no material)
}
