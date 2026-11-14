#version 450
// GPU cluster culling (documentation/gpu_cluster_culling.md, features/cull/xlionrender_gpu_cull.h): a thread a candidate cluster. Its
// local box through its draw's local to clip; in the view, a command and a record appended to its run (and to its prepass run): the
// scene pass draws each run with one DrawIndexedIndirectCount (the record: xgeom_static_mb_clusters.vert XLION_RECORDS, by firstInstance).
// Two passes, nothing a frame late: pass 1 (mode 0) keeps the clusters seen last frame (their visibility: Vis); the depth they lay becomes
// a Hi-Z pyramid; pass 2 (mode 1) tests every cluster in the view against it - its visibility for the next frame - and keeps the visible
// ones pass 1 did not draw. Mode 2: frustum only, every cluster in the view (no occlusion)
layout(local_size_x = 64) in;

struct candidate
{
    uvec4   Runs;           // its run, the run's first command, its prepass run (0xFFFFFFFF: none), that run's first command
    uvec4   Draw;           // its draw's matrix, the command: index count, first index, vertex offset
    uvec4   Rec0;           // the record: cluster, instance, GI offset, GI level
    uvec4   Rec1;           //   material base, GI vertex delta; its visibility's entry; w: its LOD group | its LOD << 28 (0xFFFFFFFF: no LOD test)
    vec4    Center;         // its box (local)
    vec4    Half;
};
// (A39 S1, 2026-11-14: the draw's matrix from the instance buffer the vertex shader reads - nothing a draw uploaded a frame; the view in
// the push constants. Rows: local -> the instance buffer's origin, xlionrender_plan_forward.h instance_data)
struct instance_t { vec4 L2W[3]; vec4 Material[3]; vec4 Prev[3]; };
// (A39 S3: a LOD group - a submesh's LOD 0 box, its LODs' errors (object space, k = 1..8; 1e30: none), its LOD count and finest resident;
// the shader picks its LOD as cluster_geom::PickLOD did on the CPU, each LOD's clusters are candidates and keep only the picked one)
struct lod_group { vec4 Center; vec4 Half; vec4 Err[2]; uvec4 Info; };
layout(std430, set = 0, binding = 0) readonly  buffer cand_b    { candidate C[]; };
layout(std430, set = 0, binding = 1) readonly  buffer inst_b    { instance_t Inst[]; };
layout(std430, set = 0, binding = 2)           buffer count_b   { uint Counts[]; };     // pass 1's, then (at m_RunsCap) pass 2's; at 2 m_RunsCap: the triangles shaded,
                                                                                         // then the triangles in the view and those pass 2 hid (adaptive occlusion)
layout(std430, set = 0, binding = 3) writeonly buffer cmd_b     { uint Cmd[]; };        // VkDrawIndexedIndirectCommand: 5 each
layout(std430, set = 0, binding = 4) writeonly buffer rec_b     { uint Rec[]; };        // the vertex shader's DrawRecord: 6 each
layout(std430, set = 0, binding = 5)           buffer vis_b     { uint Vis[]; };        // 1: visible last frame
layout(std430, set = 0, binding = 6) readonly  buffer hiz_b     { float HiZ[]; };       // xlionrender_hiz_comp.glsl
layout(std430, set = 0, binding = 7) readonly  buffer group_b   { lod_group Groups[]; };

layout(push_constant) uniform push_t
{
    uint m_First, m_Count, m_GroupBase, m_Mode; // the candidates and LOD groups in use; 0: pass 1, 1: pass 2, 2: frustum only
    uint m_Second, m_RunsCap;                   // pass 2's commands: after pass 1's (m_Second on); its counts: m_RunsCap on
    uint m_W, m_H;                              // Hi-Z level 0's size (the view's)
    mat4 m_View;                                // the instance buffer's origin to clip
    vec4 m_Eye;                                 // the eye from that origin; w: pixels per unit at distance 1 / the LOD's pixels (0: no LOD test)
} Push;

void Append(uint Run, uint First, candidate K, uvec4 R0, uvec2 R1, uint Pass)
{
    const uint o = First + (Pass == 1u ? Push.m_Second : 0u) + atomicAdd(Counts[Run + (Pass == 1u ? Push.m_RunsCap : 0u)], 1u);
    Cmd[o * 5u + 0u] = K.Draw.y; Cmd[o * 5u + 1u] = 1u; Cmd[o * 5u + 2u] = K.Draw.z; Cmd[o * 5u + 3u] = K.Draw.w; Cmd[o * 5u + 4u] = o;
    Rec[o * 6u + 0u] = R0.x; Rec[o * 6u + 1u] = R0.y; Rec[o * 6u + 2u] = R0.z; Rec[o * 6u + 3u] = R0.w; Rec[o * 6u + 4u] = R1.x; Rec[o * 6u + 5u] = R1.y;
}

// Whether the box (clip space: center, half axes) is behind what the Hi-Z holds: its nearest depth past the farthest depth of the texels
// it covers, at the level where it covers at most 2 x 2. Crossing the near plane: never
bool Occluded(vec4 Cc, vec4 Ax, vec4 Ay, vec4 Az)
{
    vec3 Lo = vec3(1e30), Hi = vec3(-1e30);
    for (int i = 0; i < 8; ++i)
    {
        const vec4 P = Cc + ((i & 1) != 0 ? Ax : -Ax) + ((i & 2) != 0 ? Ay : -Ay) + ((i & 4) != 0 ? Az : -Az);
        if (P.w <= 1e-6 || P.z <= 0.0) return false;
        const vec3 N = P.xyz / P.w;
        Lo = min(Lo, N); Hi = max(Hi, N);
    }
    const vec2  Size = vec2(Push.m_W, Push.m_H);
    const ivec2 Last = ivec2(Push.m_W, Push.m_H) - 1;
    const ivec2 P0   = clamp(ivec2(floor((Lo.xy * 0.5 + 0.5) * Size)), ivec2(0), Last);
    const ivec2 P1   = clamp(ivec2(floor((Hi.xy * 0.5 + 0.5) * Size)), ivec2(0), Last);
    const ivec2 Q0 = P0 >> 1, Q1 = P1 >> 1;             // (level 0: half the view's size - xlionrender_hiz_comp.glsl)
    uint Offset = 0u, W = (Push.m_W + 1u) / 2u, H = (Push.m_H + 1u) / 2u;
    int  L = 0;
    while (any(greaterThan((Q1 >> L) - (Q0 >> L), ivec2(1))) && (W > 1u || H > 1u))
    {
        Offset += W * H; W = (W + 1u) / 2u; H = (H + 1u) / 2u; ++L;
    }
    const ivec2 T0 = Q0 >> L, T1 = Q1 >> L;
    float Far = 0.0;
    for (int y = T0.y; y <= T1.y; ++y)
        for (int x = T0.x; x <= T1.x; ++x)
            Far = max(Far, HiZ[Offset + uint(y) * W + uint(x)]);
    return Lo.z > Far;
}

void main()
{
    const uint i = gl_GlobalInvocationID.x;
    if (i >= Push.m_Count) return;
    const candidate K = C[Push.m_First + i];
    const instance_t I = Inst[K.Rec0.y];
    const mat4 L2W = transpose(mat4(I.L2W[0], I.L2W[1], I.L2W[2], vec4(0, 0, 0, 1)));
    const mat4 M = Push.m_View * L2W;

    // Its LOD group's pick (cluster_geom::PickLOD): the coarsest LOD whose error, the draw's scale larger, stays under the pixels at the
    // distance from the eye to the group's box; clamped to the finest resident and the last; only that LOD's clusters go on
    if (K.Rec1.w != 0xFFFFFFFFu && Push.m_Eye.w > 0.0)
    {
        const lod_group G = Groups[Push.m_GroupBase + (K.Rec1.w & 0x0FFFFFFFu)];
        const vec3  Cw = (L2W * vec4(G.Center.xyz, 1.0)).xyz;
        const vec3  Hw = abs(L2W[0].xyz) * G.Half.x + abs(L2W[1].xyz) * G.Half.y + abs(L2W[2].xyz) * G.Half.z;
        const float Dist  = length(clamp(Push.m_Eye.xyz, Cw - Hw, Cw + Hw) - Push.m_Eye.xyz);
        const float Scale = max(max(length(L2W[0].xyz), length(L2W[1].xyz)), length(L2W[2].xyz));
        const int   n = int(G.Info.x);
        int k = 0;
        while (k + 1 < n && k < 8 && G.Err[(k) >> 2][(k) & 3] * Scale * Push.m_Eye.w <= Dist) ++k;     // (Err[j] is LOD j + 1's)
        const int Eff = clamp(max(k, int(G.Info.y)), 0, n - 1);
        if (uint(Eff) != (K.Rec1.w >> 28))
        {
            if (Push.m_Mode == 1u) Vis[K.Rec1.z] = 0u;      // (not drawn this frame: pass 1 next frame does not take it as seen)
            return;
        }
    }

    // The box in clip space: its center and half axes; outside a clip plane P when its nearest corner is: dot(P, center) + sum |dot(P, axis)| < 0
    const vec4 Cc = M * vec4(K.Center.xyz, 1.0);
    const vec4 Ax = M[0] * K.Half.x, Ay = M[1] * K.Half.y, Az = M[2] * K.Half.z;
    const vec4 Planes[6] = vec4[6](vec4(1, 0, 0, 1), vec4(-1, 0, 0, 1), vec4(0, 1, 0, 1), vec4(0, -1, 0, 1), vec4(0, 0, 1, 0), vec4(0, 0, -1, 1));   // Vulkan's clip volume: z in [0, w]
    bool bIn = true;
    for (int p = 0; p < 6; ++p)
        if (dot(Planes[p], Cc) + abs(dot(Planes[p], Ax)) + abs(dot(Planes[p], Ay)) + abs(dot(Planes[p], Az)) < 0.0) bIn = false;

    // The view's triangles (adaptive occlusion: the share it hides, and whether the view has enough to be worth it) - counted once a frame:
    // by pass 2, or by the frustum-only pass
    if (bIn && Push.m_Mode != 0u) atomicAdd(Counts[2u * Push.m_RunsCap + 1u], K.Draw.y / 3u);

    uint Pass = 0u;
    if (Push.m_Mode == 0u)
    {
        if (!bIn || Vis[K.Rec1.z] == 0u) return;
    }
    else if (Push.m_Mode == 1u)
    {
        const bool bOccluded = bIn && Occluded(Cc, Ax, Ay, Az);
        const bool bVisible  = bIn && !bOccluded;
        if (bOccluded) atomicAdd(Counts[2u * Push.m_RunsCap + 2u], K.Draw.y / 3u);
        const bool bWas     = Vis[K.Rec1.z] != 0u;
        Vis[K.Rec1.z] = bVisible ? 1u : 0u;
        if (!bVisible || bWas) return;          // (bWas and in the view: pass 1 drew it)
        Pass = 1u;
    }
    else if (!bIn) return;

    Append(K.Runs.x, K.Runs.y, K, K.Rec0, K.Rec1.xy, Pass);
    atomicAdd(Counts[2u * Push.m_RunsCap], K.Draw.y / 3u);         // (the stats: the triangles shaded)
    if (K.Runs.z != 0xFFFFFFFFu) Append(K.Runs.z, K.Runs.w, K, uvec4(K.Rec0.xy, 0u, 0xFFFFFFFFu), uvec2(K.Rec1.x, 0u), Pass);     // (the prepass: no GI; the material's base for the tile pass's id, XLION_TILE)
}
