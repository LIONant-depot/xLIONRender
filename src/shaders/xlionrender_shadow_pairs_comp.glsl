#version 450
// The sun's page/caster pairs (features/shadows/xlionrender_shadow_pairs.h; documentation/lighting_strategy.md P1c): a thread a candidate
// cluster. Its local box through its draw's local to light; the pages to render this frame of its kind (static casters into static pages,
// movers into mover pages) whose light-space rect it overlaps - the CPU path's test (shadows::RectOverlaps), per cluster - get a record
// each (cluster, slot, page); one indexed indirect command draws the cluster that many instances (xlionrender_shadow_pairs_vert.glsl reads
// the record by gl_InstanceIndex), appended to its run (a geometry: one DrawIndexedIndirectCount). A local candidate (a point or spot
// light's caster: its draw's matrix is its local -> world) pairs with the local lights' pages (Rect.w 2) whose frustum its box overlaps
// in clip space - the sides and the far plane, not the near one (the CPU path's frustum::Overlaps(Box, false), per cluster)
layout(local_size_x = 64) in;

struct candidate
{
    uvec4   A;              // its run, the run's first command, its cluster, its instance slot
    uvec4   B;              // the command: index count, first index, vertex offset; its draw's matrix | 0x80000000: a mover, 0x40000000: local
    vec4    Center;         // its box (local)
    vec4    Half;
};
struct page
{
    mat4    W2C;            // world -> the page's clip space (the vertex shader's)
    vec4    Atlas;          // (the vertex shader's)
    vec4    Rect;           // its light-space rect: min x, min y, size; w: 1 a mover page, 2 a local light's page
};
layout(std430, set = 0, binding = 0) readonly  buffer cand_b    { candidate C[]; };
layout(std430, set = 0, binding = 1) readonly  buffer mat_b     { mat4 L2L[]; };        // a draw's local -> light (rows: the light's X, Y, Z); local: -> world
layout(std430, set = 0, binding = 2) readonly  buffer page_b    { page Pages[]; };
layout(std430, set = 0, binding = 3)           buffer count_b   { uint Counts[]; };     // a count a run; at m_RunsCap: the records used, the pairs
                                                                                         // dropped (past the cap), the triangles drawn
layout(std430, set = 0, binding = 4) writeonly buffer cmd_b     { uint Cmd[]; };        // VkDrawIndexedIndirectCommand: 5 each
layout(std430, set = 0, binding = 5) writeonly buffer rec_b     { uint Rec[]; };        // a pair: cluster, slot, page (3 each)

layout(push_constant) uniform push_t
{
    uint m_First, m_Count, m_MatBase, m_PageBase;   // this frame's candidates, matrices, pages
    uint m_nPages, m_RecordCap, m_RunsCap, m_Pad;
} Push;

// Whether the candidate draws into page p: the sun's, its light-space box against the page's rect (of its kind); a local light's, its
// box through the page's world -> clip against the sides and the far plane
bool Pairs(candidate K, mat4 M, bool bLocal, float bMover, vec2 Lo, vec2 Hi, uint p)
{
    const vec4 R = Pages[p].Rect;
    if (!bLocal) return R.w == bMover && Hi.x >= R.x && Lo.x <= R.x + R.z && Hi.y >= R.y && Lo.y <= R.y + R.z;
    if (R.w != 2.0) return false;
    const mat4 C  = Pages[p].W2C * M;
    const vec4 Cc = C * vec4(K.Center.xyz, 1.0);
    const vec4 Ax = C[0] * K.Half.x, Ay = C[1] * K.Half.y, Az = C[2] * K.Half.z;
    const vec4 Planes[5] = vec4[5](vec4(1, 0, 0, 1), vec4(-1, 0, 0, 1), vec4(0, 1, 0, 1), vec4(0, -1, 0, 1), vec4(0, 0, -1, 1));
    for (int i = 0; i < 5; ++i)
        if (dot(Planes[i], Cc) + abs(dot(Planes[i], Ax)) + abs(dot(Planes[i], Ay)) + abs(dot(Planes[i], Az)) < 0.0) return false;
    return true;
}

void main()
{
    const uint i = gl_GlobalInvocationID.x;
    if (i >= Push.m_Count) return;
    const candidate K = C[Push.m_First + i];
    const mat4  M      = L2L[Push.m_MatBase + (K.B.w & 0x3FFFFFFFu)];
    const float bMover = (K.B.w & 0x80000000u) != 0u ? 1.0 : 0.0;
    const bool  bLocal = (K.B.w & 0x40000000u) != 0u;

    // The box in light space: its center and half extents (the absolute matrix)
    const vec3 Lc = (M * vec4(K.Center.xyz, 1.0)).xyz;
    const vec3 Lh = abs(M[0].xyz) * K.Half.x + abs(M[1].xyz) * K.Half.y + abs(M[2].xyz) * K.Half.z;
    const vec2 Lo = Lc.xy - Lh.xy, Hi = Lc.xy + Lh.xy;

    uint n = 0u;
    for (uint p = 0u; p < Push.m_nPages; ++p) n += Pairs(K, M, bLocal, bMover, Lo, Hi, Push.m_PageBase + p) ? 1u : 0u;
    if (n == 0u) return;
    const uint Base = atomicAdd(Counts[Push.m_RunsCap], n);
    if (Base + n > Push.m_RecordCap) { atomicAdd(Counts[Push.m_RunsCap + 1u], n); return; }
    uint k = Base;
    for (uint p = 0u; p < Push.m_nPages; ++p)
    {
        if (!Pairs(K, M, bLocal, bMover, Lo, Hi, Push.m_PageBase + p)) continue;
        Rec[k * 3u + 0u] = K.A.z; Rec[k * 3u + 1u] = K.A.w; Rec[k * 3u + 2u] = Push.m_PageBase + p;
        ++k;
    }
    const uint o = K.A.y + atomicAdd(Counts[K.A.x], 1u);
    Cmd[o * 5u + 0u] = K.B.x; Cmd[o * 5u + 1u] = n; Cmd[o * 5u + 2u] = K.B.y; Cmd[o * 5u + 3u] = K.B.z; Cmd[o * 5u + 4u] = Base;
    atomicAdd(Counts[Push.m_RunsCap + 2u], n * (K.B.x / 3u));
}
